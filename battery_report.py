import urllib2
import requests
import time
import datetime
import sched
import os
from sys import argv

interval = -1

def read_sensors():
    
    battery = -1

    cookie_file = open('cookie.txt')
    sessID = ""
    
    for line in cookie_file:
        if 'ZWAYSession' in line:
            sessID = line.split()[6]

    cookie = {'ZWAYSession':sessID}

    r = requests.get('http://localhost:8083/ZWaveAPI/Run.devices[9].instances[0].Battery.data.last.value', cookies=cookie).text
    battery = float(r)

    return battery

def report(s):
    
    _date = time.strftime('%m/%d/%Y')
    _time = time.strftime('%H:%M:%S')
    timestamp = {}
    timestamp['date'] = _date
    timestamp['time'] = _time 
    #timestamp = int(time.time())
    battery = read_sensors()
    print timestamp['date']+" "+timestamp['time']
    print "Battery level = %f" % battery
    
    s.enter(interval, 1, report, (s,))

def generate_session():
    os.system('''curl -i -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d '{"form": true, "login": "admin", "password": "batman911", "keepme": false, "default_ui": 1}' 10.0.0.50:8083/ZAutomation/api/v1/login -c cookie.txt''')

if __name__ == '__main__':
	
	interval = float(argv[1])
	
	generate_session()
	s = sched.scheduler(time.time, time.sleep)
	s.enter(interval, 1, report, (s,))
	s.run()
