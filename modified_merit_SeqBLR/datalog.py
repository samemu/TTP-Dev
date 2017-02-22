#!/usr/bin/env python
# Filename:     datalog.py
# Authors:      apadin, yabskbd, mjmor, dvorva
# Start Date:   5/9/2016

"""
Driver for collecting data from ZWay server and saving it to given
location. This program also maintains a separate log file for providing
device information so that the data can be analyzed later

- Adrian Padin, 1/20/2017
"""

#==================== LIBRARIES ====================#
import sys
import time
import datetime as dt
import csv
import zway


#==================== FUNCTIONS ====================#

def get_all_data(server):
    """
    Accepts a zway.Server object and returns data for all connected devices.
    """
    return [server.get_data(id) for id in server.device_IDs()]


#==================== CLASSES ====================#
    
class Datalog(object):
    """Wrapper for reading and writing sequential data to CSV files"""
    
    def __init__(self, prefix, header):
        """
        Specify the prefix of the files you want to write.
        - prefix: The files will have the format "prefix_YYYY-MM-DD.csv"
        - header: List of names of each column (exclude "timestamp")
        """
        self.prefix = prefix
        self.header = header
        self.header.insert(0, "timestamp")
        self.last_fname = None
        
    def log(self, sample, timestamp=time.time()):
        """
        Add a new sample to the correct file.
        - sample: List of values to be written (excluding timestamp)
        """
        date = dt.date.fromtimestamp(timestamp)
        fname = "{}_{}.csv".format(self.prefix, date)

        # If the file does not exist, make a new one
        try:
            with open(fname, 'rb') as fh:
                pass
        except IOError:
            with open(fname, 'wb') as fh:
                csv.writer(fh).writerow(self.header)

        # Record the sample
        sample.insert(0, timestamp)
        with open(fname, 'ab') as fh:
            csv.writer(fh).writerow(sample)
                
    def read_range(self, start, end):
        pass
        
        
def main(argv):
    """Connect to server and start the logging process."""
    host     = argv[1]
    port     = argv[2]
    prefix   = argv[3]
    username = ""
    password = ""
    if len(argv) > 5:
        username = argv[4]
        password = argv[5]
    server   = zway.Server(host, port, username=username, password=password)
    device_list = server.device_IDs()
    log      = Datalog(prefix, device_list)
    
    # Timing procedure
    granularity = 10
    goal_time = int(time.time())

    while(True):
        
        while goal_time > time.time():
            time.sleep(0.2)
        goal_time = goal_time + granularity
        print "sample at time", dt.datetime.fromtimestamp(goal_time)
        
        log.log(get_all_data(server), goal_time)

if __name__ == '__main__':
    main(sys.argv)
    


