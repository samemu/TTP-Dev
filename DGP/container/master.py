#!/usr/bin/env python2

import socket
import time

UDP_IP = "172.18.1.1"
UDP_PORT = 8088

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

connected = False
while not connected:
    try:
        sock.bind((UDP_IP, UDP_PORT))
        connected = True
    except:
        pass

while True:
    data, addr = (sock.recvfrom(1024))

    curr_file = open('data.csv', 'a')
    timestamp = str(int(time.time()))
    curr_file.write(timestamp + "," + addr[0] + "," + data + "\n")
    curr_file.flush()
    curr_file.close()
