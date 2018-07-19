#!/usr/bin/env python2

import docker
import os
from time import time
import pickle

from preparer import gen_features, read_json_from_file

docker_client = docker.from_env()

print "Generate feature data..."
# gen_features("container/basis.dat")
with open("/home/markzz/Documents/Data-Modeling/matrix", "rb") as f:
    a = pickle.load(f)

with open("/home/markzz/Documents/Data-Modeling/sklearn", "rb") as f:
    b = pickle.load(f)

with open("container/matrix", "wb") as f:
    pickle.dump(a, f)

with open("container/sklearn", "wb") as f:
    pickle.dump(b, f)

print "Setting up network..."
ipam_pool = docker.types.IPAMPool(
    subnet='172.18.0.0/16'
)

ipam_config = docker.types.IPAMConfig(
    pool_configs=[ipam_pool]
)

net = docker_client.networks.create('dgp', driver='bridge', ipam=ipam_config)

print "Setting up Docker image..."
path = os.path.dirname(os.path.abspath(__file__)) + "/container"
tag = 'dgp-' + str(int(time() * 100))
docker_client.images.build(path=path, tag=tag)

print "Starting master container..."
container = docker_client.containers.run(tag, 'python master.py', detach=True)
net.connect(container, "172.18.1.1")

print "Starting messenger containers..."
conf = read_json_from_file('conf.json')
print conf
for key, value in conf[u'buildings'][0].iteritems():
    for x in range(0, value):
        docker_client.containers.run(tag, 'python messenger.py "' + key + '"' + str(conf[u"delay"]), network='dgp', detach=True)



