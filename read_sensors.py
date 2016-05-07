import urllib2
import requests
import time
import datetime
import sched
import os
from sys import argv

interval = -1

def read_sensors():
    
    url = 'http://10.0.0.50:8083/fhem?room=EnOcean'
    response = urllib2.urlopen(url)
    
    power = -1
    door = -1
    temp = -1
    switch = -1
    light = -1
    humidity = -1
    motion = -1

    for line in response:
        #print line
        if 'EnO_contact_01812FA4' in line and 'class="col2"' in line:
            tokens = line.split('>')
            state = tokens[2][0:-5]
            if state=='open':
                door=1
            elif state=='closed':
                door=0
            else:
                door=-1
    
        if 'EnO_switch_002A00C8' in line and 'class="col2"' in line:
            tokens = line.split('>')
            state = tokens[2][0:-5]
            if state=='A0':
                switch=0
            elif state=='AI':
                switch=1
            else:
                switch=-1
    
        #if 'EnO_sensor_01831879' in line and 'class="col2"' in line:
        #    tokens = line.split('>')
        #    state = tokens[2][0:-5]
        #    temp = float(state)

    cookie_file = open('cookie.txt')
    sessID = ""
    
    for line in cookie_file:
        if 'ZWAYSession' in line:
            sessID = line.split()[6]

    cookie = {'ZWAYSession':sessID}

    r = requests.get('http://localhost:8083/ZWaveAPI/Run.devices[9].instances[0].SensorMultilevel.data[1].val.value', cookies=cookie).text
    temp = float(r)

    r = requests.get('http://localhost:8083/ZWaveAPI/Run/devices[6].instances[0].SensorMultilevel.data[4].val.value', cookies=cookie).text
    power = float(r)

    r = requests.get('http://localhost:8083/ZWaveAPI/Run.devices[9].instances[0].SensorMultilevel.data[3].val.value', cookies=cookie).text
    light = float(r)

    r = requests.get('http://localhost:8083/ZWaveAPI/Run/devices[7].instances[0].SensorMultilevel.data[5].val.value', cookies=cookie).text
    humidity = float(r)
    
    r = requests.get('http://localhost:8083/ZWaveAPI/Run/devices[9].instances[0].SensorBinary.data[1].level.value', cookies=cookie).text
    if (r == 'true'):
        motion = 1
    else:
        motion = 0

    return (temp, door, switch, power, light, humidity, motion)

def report(s):
    
    _date = time.strftime('%m/%d/%Y')
    _time = time.strftime('%H:%M:%S')
    timestamp = {}
    timestamp['date'] = _date
    timestamp['time'] = _time 
    #timestamp = int(time.time())
    (temp, door, switch, power, light, humidity, motion) = read_sensors()
    print timestamp['date']+" "+timestamp['time']
    print "Door = %d" % door
    print "Temperature = %.1f deg C" % temp
    print "Switch = %d" % switch
    print "Power = %.3f W" % power
    print "Light = %d" % light
    print "Humidity = %d" % humidity
    print "Motion = %d" % motion
    
    line = timestamp['date']+","+timestamp['time']
    line += ","+str(power)
    line += ","+str(door)
    line += ","+str(temp)
#    line += ","+str(switch)
    line += ","+str(int(light))
    line += ","+str(int(humidity))
    line += ","+str(motion)
    line += "\n"

    print line
    
    file = open('./EnOceanData.csv', 'a')
    file.write(line)

    s.enter(interval, 1, report, (s,))

def generate_session():
    os.system('''curl -i -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d '{"form": true, "login": "admin", "password": "batman911", "keepme": false, "default_ui": 1}' 10.0.0.50:8083/ZAutomation/api/v1/login -c cookie.txt''')

if __name__ == '__main__':
	
	interval = float(argv[1])
	
	generate_session()
	s = sched.scheduler(time.time, time.sleep)
	s.enter(interval, 1, report, (s,))
	s.run()
