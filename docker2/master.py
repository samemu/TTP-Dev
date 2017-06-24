import socket
import time
import pandas as pd

UDP_IP = "172.18.1.1"
UDP_PORT = 8088

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# dictionary mapping IP address of messenger to corresponding messenger data file
ip_dict = {}

while True:
    data, addr = (sock.recvfrom(1024))
    # new messenger? add IP to dict, make new data file
    if addr[0] not in ip_dict:
        ip_dict[addr[0]] = './data_' + str(len(ip_dict)) + '.csv'
        curr_file = open(ip_dict[addr[0]], 'w')
    else:
        # append to data file
        curr_file = open(ip_dict[addr[0]], 'a')

    timestamp = str(int(time.time()))
    curr_file.write(timestamp + "," + addr[0] + "," + data + "\n")
    # curr_file.flush()
    curr_file.close()
