#!/bin/bash

# A warning to Windows users:
# Windows likes to be different and in that, they end their files with
# a different line ending than your UNIX counterparts. When you use this
# with a building names file, make completely sure that you are using UNIX
# line endings instead of using Windows line endings (\n instead of \r\n).
# Try using `dos2unix` before running this.

IFS=$'\n'
set -f
for x in `cat BuildingNames.csv`; do
    echo "STARTING CONTAINER FOR ${x}"
    docker run --net mynet -d quick_start python messenger.py "${x}"
done