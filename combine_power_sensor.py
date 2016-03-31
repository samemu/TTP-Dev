
sensor_in = open('./data/EnOceanCSV.csv', 'r')
power_in = open('./data/wattson_csv.csv', 'r')
combined_out = open('./data/Power_sensor_combined.csv', 'w')

if __name__ == '__main__':
    for line in sensor_in:
        timestamp = line.split(',')[0]
        power = 0
        for pline in power_in:
            pstamp = pline.split(',')[0]
            ptime = int(pstamp)
            stime = int(timestamp)
            
            if abs(ptime - stime) < 60:
                print "power for stamp: %s found" % timestamp
                power = pline.split(',')[1].strip()
                break
        
        line_out = line.split(',')[0]+','+power
        for col in line.split(',')[1:]:
            line_out += ','+col
        
        combined_out.write(line_out)
        print line_out