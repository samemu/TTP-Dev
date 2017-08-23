from sys import argv
import pickle
from random import randint
import socket
import time

import matplotlib.pylab as m
import numpy as n

from preparer import read_json_from_file

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_ip = '172.18.1.1'
    udp_port = 8088

    print "DEBUG :: Reading file..."
    with open('basis.dat', 'rb') as f:
        data = pickle.load(f)

    print "DEBUG :: Sending data..."
    model = read_json_from_file('modeling_coeffs_output.txt')
    index = randint(0, len(model['building_types'][argv[1]]) - 1)

    const = model['building_types'][argv[1]][index]
    sock.sendto(argv[1], (udp_ip, udp_port,))

    v = []
    for x in data:
        val = 0
        for y in range(0, len(const)):
            val += x[y] * const[y]

        v.append(val)
        # sock.sendto(str(val), (udp_ip, udp_port,))
        # print "DEBUG :: Sent %d..." % (val,)
        # time.sleep(0.5)

    x = n.linspace(1, len(v), len(v))
    m.plot(x, v)
    m.show()