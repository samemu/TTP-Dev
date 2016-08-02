from __future__ import division
import numpy as np
import scipy as sp
import sys, getopt
from datetime import date
import subprocess
import urllib2
import time
import random
from numpy import genfromtxt
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil, os, sys
from datetime import datetime
from datetime import date
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

startTime = datetime.now()
timestamp = str(datetime.today().year)+"-"+str(datetime.today().month)+"-"+str(datetime.today().day)
global dataReadTime
dataReadTime = datetime.now()


#Reading in umass3 data
#OBSERVS_PER_HR = 60
#training_size = 48
#training = training_size * OBSERVS_PER_HR
#tempdata =  np.loadtxt('./data/umass_01min_allcircuits_homeA_v3.dat', delimiter=",")
#times = tempdata[:24*OBSERVS_PER_HR,0]
#tempdata = tempdata[:24*OBSERVS_PER_HR,1:] # exclude time
#tempdata = preprocessing.normalize(tempdata, norm='max', axis=0)
#data = tempdata[:training]
#test_data = tempdata


#Reading in Atman's apartment data
tempdata = genfromtxt('data/Power_sensor_combined.csv', delimiter=',')
Times = tempdata[:,0]
tempdata = tempdata[:,1:]
shiftnorm = np.max(np.abs(tempdata[:,0]))
tempdata = preprocessing.normalize(tempdata, norm='max', axis=0) #Do not alter without changing shift
training = 4000
data = tempdata[:training, :]
test_data = tempdata



x_train = np.array([ i[1::] for i in data])
x_test = np.array([ i[1::] for i in test_data])

y_train = np.array([(i[0]) for i in data])
y_train = y_train[np.newaxis]
y_train = 1.*y_train.T

y_test = np.array([(i[0]) for i in test_data])
y_test = y_test[np.newaxis]
y_test = 1.*y_test.T
y_target_real = y_test

M = len(x_train[0]) #numb of features

# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 1

# Network Parameters
n_hidden = int(np.floor(M/2)) # 1st layer num features
n_input = M 
n_output = 1 


#Function to build TensorBoard summaries
def variable_summaries(var, name):
  with tf.name_scope("summaries"):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

##Create a temp directory for Summary info
TMPDir='./tmplog'
try:
    shutil.rmtree(TMPDir)
except:
    print "Tmp Dir did not exist, creating new dir"
os.mkdir(TMPDir, 0755 )


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])


#hidden layer
W_hidden= tf.Variable(tf.truncated_normal([n_input, n_hidden]))
variable_summaries(W_hidden, 'Hidden Layer Weights')

bias = tf.Variable(tf.truncated_normal([n_hidden]))
variable_summaries(bias, 'bias')

#hidden layers activation function
hidden = tf.sigmoid(tf.matmul(x, W_hidden) + bias)
tf.histogram_summary('Sigmoid Activations', hidden)

#Output Layer
W_out = tf.Variable(tf.truncated_normal([n_hidden, n_output]))
variable_summaries(W_out, 'Output Layer Weights')

output = tf.sigmoid(tf.matmul(hidden, W_out))


#Learning
cross_entropy = -(y * tf.log(output) + (1 - y) * tf.log(1-output))
#Can also use tf.square(y - output)
#tf.scalar_summary('Cross_Entropy', cross_entropy)


loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

#Define Accuracy for later
accuracy = tf.abs(y-output)

#Initialize the graph
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#Merge Summaries
summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(TMPDir + '/logs',sess.graph_def)

#Train the network in batches of 100

for i in xrange(0, len(x_train), batch_size):
    cvalues = sess.run([train, loss, W_hidden, bias, W_out], feed_dict={x: x_train[i:i+batch_size], y: y_train[i:i+batch_size]})  
    if i % 200 == 0:
        print("")
        print("step: {:>3}".format(i))
        print("loss: {}".format(cvalues[1]))
        print("b_hidden: {}".format(cvalues[3]))
        print("W_hidden: {}".format(cvalues[2]))
        print("W_output: {}".format(cvalues[4]))
        
    if i % 10 ==0:
        summary, acc = sess.run([summary_op, accuracy], feed_dict={x:x_test, y: y_test})
        summary_writer.add_summary(summary, i)
    else:
        summary, _ = sess.run([summary_op, train], feed_dict = {x:x_train[batch_size+i:batch_size*2+i], y:y_train[batch_size+i:batch_size*2+i]})
        summary_writer.add_summary(summary, i)

pred = sess.run(output, feed_dict={x:x_test})

    # Anomaly scenarios - Injecting anomalies of size shift
    #The following was taken from BLR for anomaly injection
(n_samples, total_features) = np.shape(tempdata)
y_pred = np.zeros(n_samples)
severe_array = np.zeros(n_samples)
anomaly_prediction = np.zeros(n_samples)
ground_truth = np.zeros(n_samples) # 0: no anomaly, 1: anomaly
true_positives = np.zeros(n_samples)
alert_counter = 0
p_array = []
Tp = 0
Fp = 0
Fn = 0
delays = []  
excluded_times = [] 

#Anomaly Information
p = 3.0 #Standard deviations from sample mean
shift = 3000           
EVENT_NUMS = 2         
DURATION = 100  
events = []
event_dict = {}
real_shift = shift
shift = (1. * shift) / (1. * shiftnorm)

for k in range(EVENT_NUMS):
    t_rand = random.randint(training, n_samples - DURATION)
    print "Setting anomaly at time %d " % t_rand
    events.append(t_rand)
    anomaly_events = set(np.arange(t_rand, t_rand + DURATION))
    event_dict[k] = [t_rand, t_rand + DURATION]
    for t in anomaly_events:
        test_data[t,0] = max(0,test_data[t,0] + shift)
        ground_truth[t] = 1
        
#Setting target and prediction values        
y_target = test_data[:, 0]
y_target = y_target[np.newaxis]
y_target = 1.*y_target.T

test_data = test_data[:, 1:]
y_pred = pred

        #Anomaly based on hyperparameter of Variance B_ML^-1
inBml = (1./n_samples) * np.sum((y_pred-y_target_real)**2)
sigma = np.sqrt(inBml)
anomaly_threshold = p * sigma
error = y_target - y_pred


#Some non tensorboard plots
plt.plot(pred, label="predictions")
plt.plot(y_test, label="Target Values")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.savefig("PredictionsAndTargets.png")

for i in range(0,len(test_data)):
    if np.abs(error[i]) <= anomaly_threshold:
        anomaly_prediction[i] = 0
        alert_counter = 0
    elif np.abs(error[i]) > anomaly_threshold and alert_counter == 0:
        alert_counter = 1
        anomaly_prediction[i] = 0
    elif np.abs(error[i]) > anomaly_threshold and alert_counter == 1:
        anomaly_prediction[i] = 1
        excluded_times.append(i)
    
    
    if ground_truth[i] == 1 and anomaly_prediction[i] == 1:
            Tp += 1
            #print "True Positive %d" % (n)
            for k in range(EVENT_NUMS):
                if event_dict[k][0] <= i <= event_dict[k][1]:
                    delays.append(i-event_dict[k][0])
                    #print n, event_dict[k][0], event_dict[k][1]
                    event_dict[k][1] = i
    elif ground_truth[i] == 0 and anomaly_prediction[i] == 1:
            Fp += 1
    elif ground_truth[i] == 1 and anomaly_prediction[i] == 0:
            Fn += 1


# Hereafter is just result reporting and graphing
rmse = []
Re_mse = []
smse = []
co95 = []

# Prediction accuracy
T = n_samples-training
#y_target = data[:,0]
# Prediction Mean Squared Error (raw values)
PMSE_score = 1./T * np.linalg.norm(y_target_real[training:] - y_pred[training:])**2
confidence = 1.96 / np.sqrt(T) *  np.std(np.abs(y_target_real[training:]-y_pred[training:]))
# Relative Squared Error
Re_MSE = np.linalg.norm(y_target_real[training:]-y_pred[training:])**2 / np.linalg.norm(y_target_real[training:])**2 
# Standardise Mean Squared Error
SMSE = 1./ T * np.linalg.norm(y_target_real[training:]-y_pred[training:])**2 / np.var(y_target_real[training:]) 

rmse.append(np.sqrt(PMSE_score))
co95.append(confidence)
Re_mse.append(Re_MSE)
smse.append(SMSE)



print "--------------------------------------------------------------------------------------------------------------------------------------"
print "%20s |%25s |%20s" % ("RMSE-score (raw)", "Relative MSE", "SMSE")
print "%20.2f |%25.2f |%20.2f " % (np.mean(np.asarray(rmse)), np.mean(np.asarray(Re_mse)), np.mean(np.asarray(smse)))



print "--------------------------------------------------------------------------------------------------------------------------------------"
print "True Pos = %d, False Pos = %d, False Neg = %d" % (Tp, Fp, Fn)
if EVENT_NUMS != 0:
    print "Precision score: %.3f" % precision_score(ground_truth, anomaly_prediction)
    print "Recall score: %.3f" % recall_score(ground_truth, anomaly_prediction)
    print "F1 score: %.3f" % f1_score(ground_truth, anomaly_prediction)
    global F1
    F1 = f1_score(ground_truth, anomaly_prediction)
#    print "Mean time to detect: %.3f" % (np.mean(np.asarray(delays))) 
print "-------------------------------------------------------------------------------------------------------------------------------------"



print " "
print "--------------------------------------------------------------------------------------------------------------------------------------"
print " "
print "Time taken to run the algorithm"
print datetime.now() - startTime
print " "
print "--------------------------------------------------------------------------------------------------------------------------------------"
print " "
print " "


print "\nTo see the output, run the following:"
print "$ tensorboard --logdir='%s/logs'" %(TMPDir)
