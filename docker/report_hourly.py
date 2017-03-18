import numpy as np
import time
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dist_ip = '172.18.1.1'
dist_port = 8088

dfile = open('./data.csv', 'r')
data = np.loadtxt(dfile, delimiter=',')

for i in range(len(data)):
	current = str(data[i])
	sock.sendto(current, (dist_ip, dist_port))

	time.sleep(10)
