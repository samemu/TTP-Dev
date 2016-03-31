import time
import datetime

if __name__ == "__main__":
    file = "./data/EnOceanData.csv"
    out = "./data/EnOceanCSV.csv"
    f = open(file, 'r')
    o = open(out, 'w')

    f.readline()
    for line in f:
        tokens = line.split(',')
        d = tokens[0]
        t = tokens[1]
        dt = d+" "+t
        stamp = int(time.mktime(datetime.datetime.strptime(dt, "%m/%d/%Y %H:%M:%S").timetuple()))
        #stamp = int(datetime.timedelta(days=day,hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds())
        line_out = str(stamp)
        for token in tokens[2:]:
            line_out = line_out+','+token
        print line_out
        o.write(line_out)