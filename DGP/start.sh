#!/bin/bash

source config

echo "Generating coefficients..."
python2 preparer.py

echo "Starting master..."
./master.sh

echo "Starting messengers..."

for x in {1..${RECREATION}}; do
    docker run --net mynet -d quick_start python messenger.py "RECREATION"
done

for x in {1..${ADMIN}}; do
    docker run --net mynet -d quick_start python messenger.py "ADMIN"
done

for x in {1..${MEDICAL}}; do
    docker run --net mynet -d quick_start python messenger.py "MEDICAL"
done

for x in {1..${MUSEUM}}; do
    docker run --net mynet -d quick_start python messenger.py "MUSEUM"
done

for x in {1..${COMMONS}}; do
    docker run --net mynet -d quick_start python messenger.py "COMMONS"
done

for x in {1..${LIBRARY}}; do
    docker run --net mynet -d quick_start python messenger.py "LIBRARY"
done

for x in {1..${RESEARCH}}; do
    docker run --net mynet -d quick_start python messenger.py "RESEARCH"
done

for x in {1..${DEPARTMENT}}; do
    docker run --net mynet -d quick_start python messenger.py "DEPARTMENT"
done

for x in {1..${OTHER}}; do
    docker run --net mynet -d quick_start python messenger.py "OTHER"
done

for x in {1..${LAB}}; do
    docker run --net mynet -d quick_start python messenger.py "LAB"
done

for x in {1..${PARKING}}; do
    docker run --net mynet -d quick_start python messenger.py "PARKING"
done

for x in {1..${DORM}}; do
    docker run --net mynet -d quick_start python messenger.py "DORM"
done

for x in {1..${MEMORIAL}}; do
    docker run --net mynet -d quick_start python messenger.py "MEMORIAL"
done

for x in {1..${SPORTS}}; do
    docker run --net mynet -d quick_start python messenger.py "SPORTS"
done

for x in {1..${HOSPITAL}}; do
    docker run --net mynet -d quick_start python messenger.py "HOSPITAL"
done

for x in {1..${AUDITORIUM}}; do
    docker run --net mynet -d quick_start python messenger.py "AUDITORIUM"
done

echo "Done..."
