#!/usr/bin/env python2

import csv
import numpy as np
import datetime
import json
import pickle


import sklearn.preprocessing as preprocess

import clustering_noloop as cl
import basis_justgaussian as basis

from Class_Schedule2015 import *

SECS_IN_DAY = 3600 * 24


def is_weekend(s, e):
    tmp = s
    ret = []
    while tmp < e:
        if tmp.weekday() != 5 and tmp.weekday() != 6:
            ret.append(0)
        else:
            ret.append(1)

        tmp = tmp + datetime.timedelta(minutes=2)

    return ret


def is_class(s, e):
    tmp = s
    ret = []
    while tmp < e:
        b = False
        for x in FA_CLASS + WN_CLASS + SU_CLASS + SP_CLASS:
            ret.append(1)
            b = True

        if not b:
            ret.append(0)

        tmp = tmp + datetime.timedelta(minutes=2)

    return ret


def day_periods(per, s, e):
    c = 1440 // per
    ret = []
    tmp = s
    while tmp < e:
        time = (tmp.hour * 60) + tmp.minute
        for x in map(lambda y: y * c, range(1, per + 1)):
            if time < x:
                ret.append(x / c)
                break

        tmp = tmp + datetime.timedelta(minutes=2)

    return ret


def mean(arr):
    return float(sum(arr) / len(arr))


def get_data(feature_files):

    #Load Feature Values
    features = []
    for file_name in feature_files:
        try:
            features.append(np.loadtxt(file_name, delimiter=','))
        except:
            print('Error: You either do not have access to the data files '+
                  'or you have your data files in the wrong folder. fn %s' % (file_name,))
            exit(1)



    Diff_from_avgtemp = features[6] - features[2] #Avg_Daily_Temp - Actual_Temp
    features[6] = Diff_from_avgtemp

    return features

def correct_arrays(array, g):
    ret = []
    for x in range(0, len(array) // g):
        ret.append(mean(array[x * g:(x+1) * g]))

    return ret


def read_date(date_str):
    return datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M")


def read_json_from_file(fn):
    with open(fn) as json_file:
        ret = json.load(json_file)

    return ret


def gen_features(outfile):
    conf = read_json_from_file("conf.json")
    start = read_date(conf['start_date'])
    end = read_date(conf['end_date'])

    data = read_json_from_file("modeling_program_output.json")

    # gran = SECS_IN_DAY // data['day_periods'] // 120
    # atemps = []
    # temps = []
    # humids = []
    # with open('Temp_AvgTemp_Humid2015_dated_corrected.csv', 'rb') as temp_csv:
    #     reader = csv.reader(temp_csv, delimiter='|')
    #     for row in reader:
    #         if start <= read_date(row[0]) < end:
    #             atemps.append(float(row[2]))
    #             temps.append(float(row[1]))
    #             humids.append(float(row[3]))
    #
    # holiday = []
    # with open('Weighted_Holidays2015_dated_corrected.csv', 'rb') as holiday_csv:
    #     reader = csv.reader(holiday_csv, delimiter='|')
    #     for row in reader:
    #         if start <= read_date(row[0]) < end:
    #             holiday.append(float(row[1]))
    #
    # class_sched = is_class(start, end)
    # weekends = is_weekend(start, end)
    # periods = day_periods(data['day_periods'], start, end)
    # temp_variance = map(lambda x, y: x-y, atemps, temps)

    feature_files = ["data_features/IsClass2015.csv", \
                    "data_features/IsWeekend2015.csv", \
                    "data_features/Temperature2015.csv", \
                    "data_features/Categorical_DayPeriods_2015.csv", \
                    "data_features/Weighted_Holidays2015.csv", \
                    "data_features/Humidity2015.csv", \
                   "data_features/AvgDailyTemp2015.csv", \
                    ]

    features = get_data(feature_files)
    Xdata = np.column_stack(tuple(features))
    Xdata = preprocess.scale(Xdata)
    best_k, clusters_info = cl.k_means_clustering(100, Xdata)
    Xdata, variance = basis.basis_process(Xdata, clusters_info, best_k)
    
    # pre_data = [list(i) for i in zip(class_sched, weekends, temps, periods, temp_variance, humids, holiday)]
    # pre_data = pre.scale(pre_data)

    # k, ci = cl.k_means_clustering(100, Xdata)
    # data, _ = basis.basis_process(Xdata, ci, k)


    with open(outfile, 'wb') as f:
        pickle.dump(Xdata.tolist(), f)

    print "WROTE DATA TO FILE"


if __name__ == "__main__":
    gen_features("basis.dat")
