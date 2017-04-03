import numpy as np
import time
import sys
from datetime import datetime 
import socket
from sys import argv

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dist_ip = '172.18.1.1'
dist_port = 8088

bldgtype = int(argv[1])
if bldgtype  == 0:
	print "inavlid building type"
	sys.exit(0)

current_pw = 0
b = [1,0,0,0,0,0,0,0]


while True:
	current = 0
	cur_time = datetime.now()

	if cur_time.hour < 6:
		b[1] = 1
	elif cur_time.hour < 12:
		b[2] = 1
	elif cur_time.hour < 18:
		b[3] = 1
	if cur_time.today().weekday() >= 5:
		b[7] = 1

	print b

	file = open("./Coefficients_data/coefficients.csv","r")

	for i in range (bldgtype):
		file.readline()
	x = file.readline()
	x = x.split(',')
	x = [float(i) for i in x]
	print x

	file_2 = open("./Coefficients_data/MeansStdevsForMonth4.csv","r")
	for i in range (bldgtype):
		file_2.readline()
	y = file_2.readline()
	y = y.split(',')


	mean = float(y[1])
	stdev = float(y[2])

	print mean, stdev
	for i in range (len(b)):
		current = current + x[i]*b[i]
	current = current * np.random.normal(mean,stdev)
	
	
	sock.sendto(str(current), (dist_ip,dist_port))
	print current
	time.sleep(1)
