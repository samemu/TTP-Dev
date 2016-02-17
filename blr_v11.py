#!/Users/mgkallit/Library/Enthought/Canopy_64bit/User/bin/python
# vim:ts=4:sw=4:sts=4:tw=100
# -*- coding: utf-8 -*-
#
# Author: Michael Kallitsis, 2015-03-24
# 
# Example usage:
# ./blr_detection_v7.py --dataset=umass2 --train_win=48 --smooth_win=120 --forecast_win=30 --reg
#


from __future__ import division
#import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy as sp
import sys, getopt
import random
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from datetime import date
import matplotlib.dates as mdates
from scipy.integrate import quad
from math import gamma
from datetime import datetime
from scipy.stats import norm
import subprocess
import urllib2
import json
import socks
import socket
import time

socks.set_default_proxy(socks.SOCKS4, "127.0.0.1", 8080)
socket.socket = socks.socksocket


startTime = datetime.now()
timestamp = str(datetime.today().year)+"-"+str(datetime.today().month)+"-"+str(datetime.today().day)

# GLOBALS
debug = 0
LTGRAY = '#CCCCCC'
GRAY = '#666666'
alrt_counter = 0
global dataReadTime
dataReadTime = datetime.now()

##################################################################################
def integrand(x, lam, sigma, H):
    C2 = lambda H: np.pi / (H*gamma(2*H)*np.sin(H*np.pi))
    A = (lam*sigma)**2 / C2(H)
    return A * (2*(1-np.cos(x))*np.abs(x)**(-2*H -1)) / (lam**2 - 2*lam*(1-np.cos(x)) + 2*(1-np.cos(x)))

##################################################################################

def train(X, y):
    # This function is used for training our Bayesian model
    # Returns the regression parameters w_opt, and alpha, beta, S_N
    # needed for the predictive distribution

    Phi = X # the measurement matrix of the input variables x (i.e., features)
    t   = y # the vector of observations for the target variable

    (N, M) = np.shape(Phi)

    # Init values for  hyper-parameters alpha, beta
    alpha = 5*10**(-3)
    beta = 5
    max_iter = 100
    k = 0

    PhiT_Phi = np.dot(np.transpose(Phi), Phi)
    s = np.linalg.svd(PhiT_Phi, compute_uv=0) # Just get the vector of singular values s

    ab_old = np.array([alpha, beta])
    ab_new = np.zeros((1,2))
    tolerance = 10**-3
    while( k < max_iter and np.linalg.norm(ab_old-ab_new) > tolerance):
        k += 1
        try:
            S_N = np.linalg.inv(alpha*np.eye(M) + beta*PhiT_Phi)
        except np.linalg.LinAlgError as err:
            print  "******************************************************************************************************"
            print "                           ALERT: LinearAlgebra Error detected!"
            print "      CHECK if your measurement matrix is not leading to a singular alpha*np.eye(M) + beta*PhiT_Phi"
            print "                           GOODBYE and see you later. Exiting ..."
            print  "******************************************************************************************************"
            sys.exit(-1)

        m_N = beta * np.dot(S_N, np.dot(np.transpose(Phi), t))
        gamma = sum(beta*s[i]**2 /(alpha + beta*s[i]**2) for i in range(M))
        #
        # update alpha, beta
        #
        ab_old = np.array([alpha, beta])
        alpha = gamma /np.inner(m_N,m_N)
        one_over_beta = 1/(N-gamma) * sum( (t[n] - np.inner(m_N, Phi[n]))**2 for n in range(N))
        beta = 1/one_over_beta
        ab_new = np.array([alpha, beta])

    S_N = np.linalg.inv(alpha*np.eye(M) + beta*PhiT_Phi)
    m_N = beta * np.dot(S_N, np.dot(np.transpose(Phi), t))
    w_opt = m_N
    return (w_opt, alpha, beta, S_N)

##################################################################################

def severity_metric(error, mu, sigma, w, Sn_1):
    # This function returns the values of the EWMA control chart. It returns the
    # Sn values, as described in the paper.
    #

    if error < mu: # left-tailed
        p_value = sp.stats.norm.cdf(error, mu, sigma)
        Zt = sp.stats.norm.ppf(p_value) # inverse of cdf N(0,1)
    else: # right-tailed
        p_value = 1 - sp.stats.norm.cdf(error, mu, sigma)
        Zt = sp.stats.norm.ppf(1-p_value) # inverse of cdf N(0,1)


    if Zt > 10:
        Zt = 10
    elif Zt < -10:
        Zt = -10

    Sn = (1-w)*Sn_1 + w*Zt

    if debug:
        if np.abs(Zt) > 90:
            print "Error = %d, p-value=%.3f, Z-score=%.3f, Sn_1=%.2f, Sn=%.2f " % (error, p_value, Zt, Sn_1, Sn)
        elif np.abs(Zt) < 0.005:
            print "Error = %d, p-value=%.3f, Z-score=%.3f, Sn_1=%.2f, Sn=%.2f " % (error, p_value, Zt, Sn_1, Sn)

    return Sn, Zt

##################################################################################

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

##################################################################################

def robust_accuracy(alerts, events, duration, thr):
    # This function is just for calculating the accuracy of the algorithm
    # (especially, the accuracy when using the robust filter - see paper). It's
    # not really needed for the execution of the algorithm though.
    #
    true_positives = set()
    false_positives = set()
    event_dict = {}
    fp_events_dict = {}
    tp_events_dict = {}
    Tp = 0
    Fp = 0
    Fn = 0
    precision = recall = 0

    for k in events:
        for t in range(2*duration): #adding a grace period of len duration
            event_dict[k+t] = 1

    robust_signal = np.where(alerts[0,:] > thr, alerts[0,:], np.zeros_like(alerts[0,:]))

    for n in range(len(robust_signal)): 
        if robust_signal[n] > 0:
            if n in event_dict and n not in tp_events_dict:
                for t in range(2*duration):
                    tp_events_dict[n+t] = 1
                true_positives.add(n)
                Tp += 1

            if n not in tp_events_dict and n not in fp_events_dict:
                for t in range(duration):
                    fp_events_dict[n+t] = 1
                false_positives.add(n)
                Fp += 1
    
    Fn = len(events) - Tp
   
    if Tp > 0:
        precision = Tp / (Tp + Fp)
        recall = Tp / (Tp + Fn)

    #print precision, recall, true_positives, false_positives
    return  precision, recall
        
##################################################################################

def usage():
    print 'gp.py OPTIONS'
    print 'OPTIONS:'
    print ' --dataset=D,  D choosen from [casas, pathouse1, pathouse2, umass1, umass2]'
    print ' --train_win=T,   training window  (in hours)'
    print ' --smooth_win=S,   smoothing window (in data points)'
    print ' --forecast_win=F,  forecasting  window (in data points)'
    print ' --reg,    performs regular regression'
    print ' --autor,    performs AR1 regression'

##################################################################################

def main(argv):
    #
    # Main function starts here. After some lines of input argument parsing, the
    # sequential implementation of the algorithm begins.
    #
    
    try:
        opts, args = getopt.getopt(argv,"h", ["reg", "autor", "dataset=", "train_win=", "smooth_win=", "forecast_win="])
        if len(args) > 0 or len(argv) == 0:
            raise getopt.GetoptError('Error: wrong arguments given.')
        if len(opts) == 1:
            for opt, arg in opts:
                if opt == '-h':
                    usage()
                    sys.exit(2)
                else:
                    raise getopt.GetoptError('Error: wrong arguments given.')
        elif len(opts) != 5:
            raise getopt.GetoptError('Error: wrong arguments given.')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("--train_win"):
            training = int(arg) * OBSERVS_PER_HR
            print "Training window set to %d measurements" % (training)
        if opt in ("--dataset"):
            dataset = str(arg).strip()
            
            if dataset == "casas":
                data =  np.loadtxt('./data/measurements_15min_timestamped_motionvalues.dat', delimiter=",")
                times = data[:,0]
                data = data[:,1:-1] # exclude time and burner/cold/hot
                OBSERVS_PER_HR = 4
                
            elif dataset == "pathouse1":
                OBSERVS_PER_HR = 60
                data = np.loadtxt('./data/pathouse_60sec.dat', delimiter=",")
                times = data[:60000,0]
                data = data[:60000,1:] # exclude time
                # Add noise tp the data
                N = len(times)
                noise = np.zeros(N)
                T = N
                for t in range(N):
                    loc = 0 
                    scale = 300 
                    noise[t] = np.random.normal(loc=loc, scale=scale)
                data[:,0] += noise
                
            elif dataset == "pathouse2":
                OBSERVS_PER_HR = 60
                data = np.loadtxt('./data/pathouse_60sec.dat', delimiter=",")
                times = data[:60000,0]
                data = data[:60000,1:] # exclude time
                # Add noise tp the data
                N = len(times)
                noise = np.zeros(N)
                T = N
                for t in range(N):
                    at = t/(24*OBSERVS_PER_HR)
                    loc = (1 + t/(10*T)) * (5*np.exp(np.sin(2*np.pi*at)) + 10*np.exp(np.sin(-8*np.pi*at)))
                    scale = np.sqrt(max(loc, loc**2/4))
                    noise[t] = np.random.normal(loc=loc, scale=scale)
                data[:,0] += noise
                
            elif dataset == "umass1":
                OBSERVS_PER_HR = 12
                # This is actually a dataset with snapshots every 5 minutes,
                # with BIN averaging of 1 minute
                data = np.loadtxt('./data/umass_01min_allcircuits_homeA.dat', delimiter=",")
                times = data[:,0]
                data = data[:,1:] # exclude time
                
            elif dataset == "umass2":
                OBSERVS_PER_HR = 60
                data =  np.loadtxt('./data/umass_01min_allcircuits_homeA_v2.dat', delimiter=",")
                times = data[:3*7*24*OBSERVS_PER_HR,0]
                data = data[:3*7*24*OBSERVS_PER_HR,1:] # exclude time
                
            elif dataset == "umass3":
                OBSERVS_PER_HR = 60
                data =  np.loadtxt('./data/umass_01min_allcircuits_homeA_v3.dat', delimiter=",")
                times = data[:3*7*24*OBSERVS_PER_HR,0]
                data = data[:3*7*24*OBSERVS_PER_HR,1:] # exclude time
                #global dataReadTime
                dataReadTime = datetime.now()
                
            elif dataset == "raspberry":
                OBSERVS_PER_HR = 5
                data =  np.loadtxt('./data/RaspberryCSV.csv', delimiter=",")
                times = data[:3*7*24*OBSERVS_PER_HR,0]
                data = data[:3*7*24*OBSERVS_PER_HR,1:] # exclude time
                #global dataReadTime
                dataReadTime = datetime.now()
                
            elif dataset == "graphite":
                print "reading data from carbon database..."

                gmetrics, OBSERVS_PER_HR = getMetrics(["circuit", "1min", "RealPowerW"])
                
                init_val = gmetrics[0]  
                data = loadData(init_val)

                # Swap the timestamp and the datavalues to get time on the left side
                for i in range(0, len(data)):
                    temp = data[i][0]
                    data[i][0] = data[i][1]
                    if temp==None or temp=="None" or temp=="null":
                        data[i][1] = 0
                    else:
                        data[i][1] = temp
                
                gmetrics = gmetrics[1:]
                
                # Read the remaining sets of data
                for gmetric in gmetrics:
                    subset = loadData(gmetric)
                    for i in range(0, len(data)):
                        x = subset[i][0]
                        if x==None or x=="None" or x=="null":
                            data[i].append(0)
                        else:
                            data[i].append(x)

                # convert data to numpy array
                data = np.array(data)
                
                print "Time taken to read data: "+str(datetime.now()-startTime)
                #global dataReadTime
                dataReadTime = datetime.now()

                print str(OBSERVS_PER_HR)+" observations/hr detected"
                times = data[:3*7*24*OBSERVS_PER_HR, 0]
                data = data[:3*7*24*OBSERVS_PER_HR, 1:] # exclude time
                
            else:
                print "ERROR. Wrong dataset given"
                sys.exit(-1)
        elif opt in ("--smooth_win"):
            smooth_hours = int(arg)
            smoothing_win = smooth_hours
            print "Smoothing window set to %d time intervals" % (smoothing_win)
        elif opt in ("--forecast_win"):
            forecast_win = int(arg) 
            print "Forecast window set to %d time intervals" % (forecast_win)
        elif opt in ("--reg"):
            data = data
            (n_samples, total_features) = np.shape(data)
            print "Dataset with %d time intervals loaded" % (n_samples)
            ar1 = 0
        elif opt in ("--autor"):
            (n_samples, total_features) = np.shape(data)
            print "Dataset with %d time intervals loaded" % (n_samples)
            ar1 = 1


    ##################################################################################
    # Initialization
    ##################################################################################
    rmse = []
    rmse_smoothed = []
    Re_mse = []
    smse = []
    co95 = []
    #DURATION = 45 # This can be adjusted to create attacks of shorter or longer durations
    DURATION = 100 # This can be adjusted to create attacks of shorter or longer durations
    Sn_1 = 0
    #w, L = (.29, 3.686)
    #w, L = (.53, 3.714)
    w, L = (0.78,4.45)
    #w, L = (.84, 3.719) # EWMA parameters. Other pairs can also be used, see paper  
    #w, L = (0.84,4.2)
    #w, L = (1, 3.719)
    #w, L = (0.53, 1)
    #w, L = (0.84, 3.719)
    sigma_w = np.sqrt(w/(2-w))
    # Calculate variance of an fGN with self-similarity parameter H
    #sigma_w, est_err = np.sqrt(quad(integrand, -np.inf, np.inf, args=(w, 1, .7)))

    THRESHOLD = L * sigma_w
    #ROBUST_THRESHOLD = min(.70 * DURATION, 15)
    ROBUST_DURATION = 10 # Parameter \nu of robust filter - see paper
    ROBUST_THRESHOLD = 2 # Paramter \theta_r of robust filter - see paper
    EVENT_NUMS = 2 # Number of injected anomalies
    y_predictions = np.zeros(n_samples)
    severe_array = np.zeros(n_samples)
    Zscore_array = np.zeros(n_samples)
    anomaly_prediction = np.zeros(n_samples)
    ground_truth = np.zeros(n_samples) # 0: no anomaly, 1: anomaly
    true_positives = np.zeros(n_samples)
    Tp = 0
    Fp = 0
    Fn = 00
    alrt_counter = 0
    excluded_times = []
    p_array = []
    delays = []
    dates = [date.fromtimestamp(t) for t in times]
    Nf = np.ceil((n_samples-training)/forecast_win) + 1
    W_array = np.zeros((total_features-1, Nf))
    wj = 0

    # Anomaly scenarios - Injecting anomalies of size shift
    shift = 3000 # Can be changed to create larger/smaller anomalies
    
    print "Using %d injections of size: %d kw and length: %d intervals" % (EVENT_NUMS, shift, DURATION)
    print "Robust Threshold: theta_r=%d, v=%d" % (ROBUST_THRESHOLD, ROBUST_DURATION)
    print "w = %.3f, L = %.3f" % (w, L)
    print ""
    
    events = []
    event_dict = {}
    for k in range(EVENT_NUMS):
        t_rand = random.randint(training, n_samples - DURATION)
        print "Setting anomaly at time %d " % t_rand
        events.append(t_rand)
        anomaly_events = set(np.arange(t_rand, t_rand + DURATION))
        event_dict[k] = [t_rand, t_rand + DURATION]
        for t in anomaly_events:
            data[t,0] = max(0,data[t,0] + shift)
            ground_truth[t] = 1
    
    # Add an additional column to account for the Auto-regressive values
    if ar1:
        a = np.zeros(n_samples)
        a[1:] = data[:-1,0]
        data = np.concatenate((data,a[:,np.newaxis]), axis=1)

    y_target = data[:,0]

    # Fit regression model for the 1st training period
    X = data[:training,1:]
    y = data[:training,0]
    (N,M) = np.shape(X)
    w_opt,a_opt, b_opt, S_N = train(X,y)
    W_array[:,wj] = w_opt; wj += 1

    mu = 0; sigma = 1000
    # Sequential execution of the algorithm starts here
    for n in range(training, n_samples):
        if n % forecast_win == 0:
            training_list = list(set(range(n-training,n)) - set(excluded_times))
            X = data[training_list,1:]
            y = data[training_list,0]
            w_opt,a_opt, b_opt, S_N = train(X,y)
            W_array[:,wj] = w_opt; wj += 1

        x_n = data[n,1:]
        y_predictions[n] = max(0, np.inner(w_opt,x_n)) 
        error = (y_predictions[n]-y_target[n])
        sigma = np.sqrt(1/b_opt + np.dot(np.transpose(x_n),np.dot(S_N, x_n)))
        if sigma < 1: sigma = 1 # Catching pathogenic cases where variance (ie, sigma) gets really really small

        # Update severity metric
        mu = mu; sigma = sigma
        Sn, Zn = severity_metric(error, mu, sigma, w, Sn_1)
        severe_array[n] = Sn
        Zscore_array[n] = Zn

        
        # Robust EWMA - two-in-a-row rule applied
        if np.abs(Sn) <= THRESHOLD: 
            anomaly_prediction[n] = 0
            # Model Validation
            U, = np.random.rand(1)
            ft = sp.stats.norm.pdf(error, mu, sigma)
            p = 1 - sp.stats.norm.cdf(error, mu, sigma)
            p_array.append(p)
            alrt_counter = 0
        elif np.abs(Sn) > THRESHOLD and alrt_counter == 0:
            alrt_counter = 1
            Sn = Sn_1
            anomaly_prediction[n] = 0
        elif np.abs(Sn) > THRESHOLD and alrt_counter == 1:
            anomaly_prediction[n] = 1
            excluded_times.append(n)
            Sn = 0
            #alrt_counter = 0

        # Checking if we correctly found the anomaly or not. If yes, is a True
        # Positive (Tp). Otherwise, we count the False Positives (Fp) and False
        # Negatives (Fn)
        if ground_truth[n] == 1 and anomaly_prediction[n] == 1:
            Tp += 1
            #print "True Positive %d" % (n)
            for k in range(EVENT_NUMS):
                if event_dict[k][0] <= n <= event_dict[k][1]:
                    delays.append(n-event_dict[k][0])
                    #print n, event_dict[k][0], event_dict[k][1]
                    event_dict[k][1] = n
        elif ground_truth[n] == 0 and anomaly_prediction[n] == 1:
            Fp += 1
        elif ground_truth[n] == 1 and anomaly_prediction[n] == 0:
            Fn += 1
            #print "False Negative %d" % (n)

        Sn_1 = Sn
        
    #
    # Hereafter is just result reporting and graphing
    #
    # Prediction accuracy
    T = n_samples-training
    y_target = data[:,0]
    y_target_smoothed = movingaverage(y_target[training:], smoothing_win)
    y_predictions_smoothed = movingaverage(y_predictions[training:], smoothing_win)

    # Prediction Mean Squared Error (smooth values)
    PMSE_score_smoothed = 1/T * np.linalg.norm(y_target_smoothed-y_predictions_smoothed)**2
    # Prediction Mean Squared Error (raw values)
    PMSE_score = 1/T * np.linalg.norm(y_target[training:] - y_predictions[training:])**2
    confidence = 1.96 / np.sqrt(T) *  np.std(np.abs(y_target[training:]-y_predictions[training:]))
    # Relative Squared Error
    Re_MSE = np.linalg.norm(y_target[training:]-y_predictions[training:])**2 / np.linalg.norm(y_target[training:])**2 
    # Standardise Mean Squared Error
    SMSE = 1/ T * np.linalg.norm(y_target[training:]-y_predictions[training:])**2 / np.var(y_target[training:]) 

    rmse_smoothed.append(np.sqrt(PMSE_score_smoothed))
    rmse.append(np.sqrt(PMSE_score))
    co95.append(confidence)
    Re_mse.append(Re_MSE)
    smse.append(SMSE)

    print "--------------------------------------------------------------------------------------------------------------------------------------"
    print "%20s |%20s |%25s |%20s" % ("RMSE-score (smoothed)", "RMSE-score (raw)", "Relative MSE", "SMSE")
    print "%20.2f |%20.2f |%25.2f |%20.2f " % (np.mean(np.asarray(rmse_smoothed)), np.mean(np.asarray(rmse)), np.mean(np.asarray(Re_mse)), np.mean(np.asarray(smse)))


    print "--------------------------------------------------------------------------------------------------------------------------------------"
    print "True Pos = %d, False Pos = %d, False Neg = %d" % (Tp, Fp, Fn)
    if EVENT_NUMS != 0:
        print "Precision score: %.3f" % precision_score(ground_truth, anomaly_prediction)
        print "Recall score: %.3f" % recall_score(ground_truth, anomaly_prediction)
        print "F1 score: %.3f" % f1_score(ground_truth, anomaly_prediction)
        print "MCC: %.3f" % matthews_corrcoef(ground_truth, anomaly_prediction)
        global F1
        global MCC
        F1 = f1_score(ground_truth, anomaly_prediction)
        MCC =  matthews_corrcoef(ground_truth, anomaly_prediction)
        print "Mean time to detect: %.3f" % (np.mean(np.asarray(delays))) 
    print "--------------------------------------------------------------------------------------------------------------------------------------"
    
    ##################################################################################
    # Graphing. 
    ##################################################################################
    
    if False:

        # Create alert level chart
        alerts = np.zeros((100,n_samples+DURATION))
        for t in range(training,n_samples):
            alerts[:,t] = np.sum(anomaly_prediction[t-ROBUST_DURATION:t])

        robust_precision, robust_recall = robust_accuracy(alerts, events, DURATION, ROBUST_THRESHOLD)
        print "--------------------------------------------------------------------------------------------------------------------------------------"
        print "Robust Precision score: %.3f" %  robust_precision
        print "Robust Recall score: %.3f" % robust_recall
        print "--------------------------------------------------------------------------------------------------------------------------------------"


        plt.rc('axes', grid=False)
        plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
        textsize = 9
        left, width = 0.1, 0.8
        rect1 = [left, 0.1, width, 0.8]
        fig = plt.figure(facecolor='white')
        axescolor  = '#f6f6f6'  # the axes background color
        ax1 = fig.add_axes(rect1, axisbg=axescolor)  #left, bottom, width, height

        for t_rand in events:
            ax1.axvspan(t_rand-4*DURATION, t_rand+4*DURATION, alpha=0.3, color='red')
        ax1.set_xlim(0,n_samples)
        ax1.plot(np.where(alerts[0,:] > 0, alerts[0,:], np.zeros_like(alerts[0,:])), color = GRAY)
        ax1.set_ylabel('Alert Dashboard')
        ax1.set_ylim([0,10])
        ax1.plot(ROBUST_THRESHOLD * np.ones_like(alerts[0,:]), "r--", lw=3)
        ax1.text(0.01, 0.28, 'robust filter', va='top', transform=ax1.transAxes, fontsize=textsize)
        distance = n_samples//5
        tick_pos = [t for t in range(distance,n_samples,distance)]
        tick_labels = [dates[t] for t in tick_pos]
        ax1.set_xlim(0,n_samples)
        ax1.set_xticks(tick_pos)
        ax1.set_xticklabels(tick_labels)
        plt.savefig('./results/graphs/blr_detection_umass3_heatmap_'+timestamp+'.pdf')



        plt.rc('axes', grid=False)
        plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
        textsize = 9
        left, width = 0.1, 0.8
        rect1 = [left, 0.7, width, 0.2]
        rect2 = [left, 0.3, width, 0.4]
        rect3 = [left, 0.1, width, 0.2]
        fig = plt.figure(facecolor='white')
        axescolor  = '#f6f6f6'  # the axes background color
        ax1 = fig.add_axes(rect1, axisbg=axescolor)  #left, bottom, width, height
        ax2 = fig.add_axes(rect2, axisbg=axescolor, sharex=ax1)
        ax3  = fig.add_axes(rect3, axisbg=axescolor, sharex=ax1)
        y_target[:training] = 0
        ax1.plot((movingaverage(y_predictions, smoothing_win) - movingaverage(y_target, smoothing_win)),"r-", lw=2)
        ax1.set_yticks([-500, 0, 500])
        ax1.set_yticklabels([-.5, 0, .5])
        ax1.set_ylim(-1000, 1000)
        ax1.set_ylabel("Error (KW)")
        ax2.plot(movingaverage(y_predictions, smoothing_win),color=GRAY, lw=2, label = 'Prediction')
        ax2.plot(movingaverage(y_target, smoothing_win), "r--", label = 'Target')
        ax2.set_yticks([2000, 4000, 6000])
        ax2.set_yticklabels([2, 4, 6])
        ax2.set_ylabel("Power (KW)")
        ax2.set_xlim(0,len(y_target))
        ax2.legend(loc='upper left')
        ax3.plot(severe_array, color=GRAY, ls="--", lw = 2)
        ax3.plot(THRESHOLD*np.ones_like(severe_array), "r--")
        ax3.plot(-THRESHOLD*np.ones_like(severe_array), "r--")
        for t_rand in events:
            ax3.axvspan(t_rand-4*OBSERVS_PER_HR, t_rand+4*OBSERVS_PER_HR, alpha=0.5, color='red')
        ax3.set_ylim(-3*THRESHOLD, 3*THRESHOLD)
        ax3.set_yticks([-THRESHOLD, THRESHOLD], ["%.2f" % (-THRESHOLD), "%.2f" % THRESHOLD])
        ax3.set_xticks(tick_pos)
        ax3.set_xticklabels(tick_labels)
        ax3.set_ylabel("Q-chart Test")
        # turn off upper axis tick labels, rotate the lower ones, etc
        for ax in ax1, ax2, ax3:
            if ax!=ax3:
                for label in ax.get_xticklabels():
                    label.set_visible(False)
            else:
                for label in ax.get_xticklabels():
                    label.set_rotation(0)
                    label.set_horizontalalignment('right')
        plt.savefig('./results/graphs/blr_detection_umass3_'+timestamp+'.pdf')


        plt.rc('axes', grid=False)
        plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
        textsize = 9
        left, width = 0.1, 0.8
        rect1 = [left, 0.1, width, 0.8]
        fig = plt.figure(facecolor='white')
        axescolor  = '#f6f6f6'  # the axes background color
        ax1 = fig.add_axes(rect1, axisbg=axescolor)  #left, bottom, width, height


        #fig, ax = plt.subplots(nrows=1, ncols=1)
        p_array = np.asarray(p_array)
        hist, bin_edges = np.histogram(p_array, density=True)
        numBins = 200
        ax1.hist(p_array, numBins,color=GRAY, alpha=0.7)
        ax1.set_ylabel("P-value distribution")
        plt.savefig('./results/graphs/pvalue_distribution_under_H0_'+timestamp+'.pdf')


        plt.rc('axes', grid=False)
        plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
        textsize = 9
        left, width = 0.1, 0.8
        rect1 = [left, 0.1, width, 0.8]
        fig = plt.figure(facecolor='white')
        axescolor  = '#f6f6f6'  # the axes background color
        ax1 = fig.add_axes(rect1, axisbg=axescolor)  #left, bottom, width, height

        im1 = ax1.imshow(W_array, aspect = 'auto', interpolation='nearest', cmap=plt.cm.YlOrRd)
        # Create divider for existing axes instance
        divider = make_axes_locatable(ax1)
        # Append axes to the right of ax1, with 10% width of ax1
        cax2 = divider.append_axes("right", size="10%", pad=0.05)
        plt.colorbar(im1, cax=cax2)
        ax1.set_ylabel('Regression Weights')
        #plt.show()

        #save the Z-score array into a file
        with open('./zscores.csv', 'w') as fz:
            for n in range(n_samples):
                fz.write("%f,%f,%f\n" % (y_target[n], y_predictions[n], Zscore_array[n]))

            
##################################################################################
def loadData(name):
    # url Strings
    url_head = "http://carbon.merit.edu/render?from=00:00_20150619&until=00:00_20150801&target="
    url_tail = "&format=json"
    # connect to graphite
    response = urllib2.urlopen(url_head + name + url_tail)
    # convert to json
    subset = json.load(response)
    # convert to list
    subset = subset[0]['datapoints']
    # print name+" read "+str(len(subset))+" datapoints"
    return subset
    
# Get the names of all metrics containing every keyword in [keywords]
def getMetrics(keywords):
    # Get every metric
    #gmetrics = json.load(urllib2.urlopen("http://carbon.merit.edu/metrics/index.json"))
    gmetrics = json.load(open("./metrics/index.json"))
    oph = 60    

    # Begin searching for desired metrics
    for keyword in keywords:
        if "min" in keyword:
            mins = int(keyword[0])
            oph = int(60/mins)
        out = []
        #print ("testing: "+keyword)
        for metric in gmetrics:
            if keyword in metric:
                #print ("adding "+metric)
                out.append(metric)
        gmetrics = out
    
    # Arrange the metrics in correct order
    temp = out
    out = []
    for i in range(0, 30):
        key = "."+str(i)+"_"
        for metric in temp:
            if key in metric:
                out.append(metric)
                break;
    
    return out, oph
    
if __name__ == "__main__":
    main(sys.argv[1:])

print " "
print "--------------------------------------------------------------------------------------------------------------------------------------"
print " "
print "Time taken to run the algorithm"
print datetime.now() - startTime
print " "
print "--------------------------------------------------------------------------------------------------------------------------------------"
print " "
print " "

f = open("./results/time_"+timestamp+".csv", "a")
print >>f, str(datetime.now() - startTime) + "," + str(dataReadTime-startTime)

global F1
global MCC
f = open("./results/Accuracy_"+timestamp+".csv", "a")
print >>f, str(F1)+","+str(MCC)

#
# EOF

