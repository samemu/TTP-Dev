#!/bin/bash
echo "starting network..."
docker network create --subnet=172.18.0.0/16 mynet

