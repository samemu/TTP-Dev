import urllib2
import time
import datetime
import sched
from sys import argv

interval = -1

def read_sensors():
    
    url = 'http://10.0.0.10:8083/fhem?room=EnOcean'
    response = urllib2.urlopen(url)
    
    power = -1
    door = -1
    temp = -1
    switch = -1
    light = -1
    humidity = -1

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
    
        if 'EnO_sensor_01831879' in line and 'class="col2"' in line:
            tokens = line.split('>')
            state = tokens[2][0:-5]
            temp = float(state)
    
    r = urllib2.urlopen('http://10.0.0.14:8083/ZWaveAPI/Run/devices[2].Meter.data[2].val.value').read()
    power = float(r)

    r = urllib2.urlopen('http://10.0.0.14:8083/ZWaveAPI/Run/devices[3].SensorMultilevel.data[3].val.value').read()
    light = float(r)

    r = urllib2.urlopen('http://10.0.0.14:8083/ZWaveAPI/Run/devices[3].SensorMultilevel.data[5].val.value').read()
    humidity = float(r)

    return (temp, door, switch, power, light, humidity)

def report(s):
    
    _date = time.strftime('%m/%d/%Y')
    _time = time.strftime('%H:%M:%S')
    timestamp = {}
    timestamp['date'] = _date
    timestamp['time'] = _time 
    #timestamp = int(time.time())
    (temp, door, switch, power, light, humidity) = read_sensors()
    print timestamp['date']+" "+timestamp['time']
    print "Door = %d" % door
    print "Temperature = %.1f deg C" % temp
    print "Switch = %d" % switch
    print "Power = %.3f W" % power
    print "Light = %.6d" % light
    print "Humidity = %d" % humidity
    
    line = timestamp['date']+","+timestamp['time']
    line += ","+str(power)
    line += ","+str(door)
    line += ","+str(temp)
    line += ","+str(switch)
    line += ","+str(int(light))
    line += ","+str(int(humidity))
    line += "\n"

    print line
    
    file = open('./EnOceanData.csv', 'a')
    file.write(line)

    s.enter(interval, 1, report, (s,))

if __name__ == '__main__':
	
	interval = int(argv[1])
	
	s = sched.scheduler(time.time, time.sleep)
	s.enter(0, 1, report, (s,))
	s.run()
