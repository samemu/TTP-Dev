#!/usr/bin/env python2

import csv
import datetime
import json
import pickle

import sklearn.preprocessing as pre

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
            if read_date(x[0]) <= tmp <= read_date(x[1]):
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

    # ret = []
    # tmp = 1
    # for x in range(0, length):
    #     ret.append(tmp)
    #     if tmp == per:
    #         tmp = 1
    #     else:
    #         tmp += 1
    #
    # return ret


def mean(arr):
    return float(sum(arr) / len(arr))


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


if __name__ == "__main__":
    conf = read_json_from_file("conf.json")
    start = read_date(conf['start_date'])
    end = read_date(conf['end_date'])

    data = read_json_from_file("modeling_program_output.json")

    gran = SECS_IN_DAY // data['day_periods'] // 120

    atemps = []
    temps = []
    humids = []
    with open('Temp_AvgTemp_Humid2015_dated_corrected.csv', 'rb') as temp_csv:
        reader = csv.reader(temp_csv, delimiter='|')
        for row in reader:
            if start <= read_date(row[0]) < end:
                atemps.append(float(row[2]))
                temps.append(float(row[1]))
                humids.append(float(row[3]))

    holiday = []
    with open('Weighted_Holidays2015_dated_corrected.csv', 'rb') as holiday_csv:
        reader = csv.reader(holiday_csv, delimiter='|')
        for row in reader:
            if start <= read_date(row[0]) < end:
                holiday.append(float(row[1]))

    class_sched = is_class(start, end)
    weekends = is_weekend(start, end)
    periods = day_periods(data['day_periods'], start, end)
    temp_variance = map(lambda x, y: x-y, atemps, temps)

    pre_data = [list(i) for i in zip(class_sched, weekends, temps, periods, temp_variance, humids, holiday)]
    pre_data = pre.scale(pre_data)

    k, ci = cl.k_means_clustering(100, pre_data)
    data, _ = basis.basis_process(pre_data, ci, k)

    with open('basis.dat', 'w') as f:
        pickle.dump(data, f)

    print "WROTE DATA TO FILE"
