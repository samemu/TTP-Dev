#!/bin/bash


echo "Spinning master containers..."
#run master container, connect to mynet
# this will also run master.py within this container
docker run --net mynet --ip 172.18.1.1 -itd quick_start python master.py

