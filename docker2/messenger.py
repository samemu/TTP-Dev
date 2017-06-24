#!/usr/bin/env python2

import csv
import time
import socket
from sys import argv


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_ip = '172.18.1.1'
udp_port = 8088
building_name = None

# the current column-num/building-id we are reading from data
try:
    building_name = argv[1]
    # print("building id: " + str(bld_id))

except IndexError:
    # if cmd line is "python messenger.py" instead of
    # "python messenger.py <bld_id>"
    print("Please specify a building id")
    exit(1)

data_points = file_len('Temperature2015.csv')

# open csv files
temp_file = open('Temperature2015.csv', 'r')
is_class_file = open('IsClass2015.csv', 'r')
is_wkend_file = open('IsWeekend2015.csv', 'r')
quarter_file = open('QuarterOfDay.csv', 'r')
coefs_file = open('CoefsLinreg.csv', 'r')

# get coefficients for the current building from the coef_file
coefs_reader = csv.reader(coefs_file, delimiter=',', quoting=csv.QUOTE_NONE)
for row in coefs_reader:
    if row[0].strip() == building_name:
        matrix_X = [float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7]), float(row[8])]
        break

# set up B matrices
matrices_B = []
for x in range(0, data_points):
    q = [int(x) for x in quarter_file.readline().split(',')]
    tmp = [1, int(temp_file.readline()), int(is_class_file.readline()), int(is_wkend_file.readline()), q[0], q[1], q[2],
           q[3]]
    matrices_B.append(tmp)

# calculate value and send it on
for B in matrices_B:
    val = 0.0
    for x in range(0, len(matrix_X)):
        val += matrix_X[x] * B[x]

    # print(val)
    sock.sendto(str(val), (udp_ip, udp_port,))
    time.sleep(0.5)
