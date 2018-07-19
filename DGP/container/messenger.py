from sys import argv
import pickle
from random import randint
import socket
import json
import time

from sklearn.linear_model import LinearRegression


def read_json_from_file(fn):
    with open(fn) as json_file:
        ret = json.load(json_file)

    return ret


if __name__ == "__main__":
    print float(argv[2])
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_ip = '172.18.1.1'
    udp_port = 8088

    print "DEBUG :: Reading file..."
    with open('matrix', 'rb') as f:
        data = pickle.load(f)

    with open("sklearn", "rb") as f:
        q = pickle.load(f)

    a = q[randint(0, len(q)-1)].predict(data)
    print a

    # print "DEBUG :: Sending data..."
    # model = read_json_from_file('modeling_coeffs_output.txt')
    # i = len(model['building_types']['DORM']) - 1
    # if i >= 0:
    #     index = randint(0, i)
    # else:
    #     exit(0)
    #
    # const = model['building_types']['DORM'][index]
    # sock.sendto('DORM', (udp_ip, udp_port,))
    #
    # v = []
    for x in a:
        print str(x)
        sock.sendto(str(x), (udp_ip, udp_port,))
        time.sleep(float(argv[2]))
    #     val = 0
    #     for y in range(0, len(const)-1):
    #         val += x[y] * const[y]
    #     # print "DEBUG :: %f" % (val,)
    #     val += const[-1]
    #
    #     # print val
    #     # v.append(val)
    #     sock.sendto(str(val), (udp_ip, udp_port,))
    #     print "DEBUG :: Sent %d..." % (val,)
    #     time.sleep(2.0)
