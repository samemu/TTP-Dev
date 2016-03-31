import sqlite3

fname = './data/wattson.db'
out = open('./data/wattson_csv.csv', 'w')

def check_output():
    f = open('./data/wattson_csv.csv', 'r')
    
    print '\n\n============================================================='
    print '                     Checking Continuity                     '
    print '============================================================='
    
    line_prev = None
    i=0
    for line in f:
        i += 1
        if line_prev == None:
            line_prev = line
            continue
        
        t1 = int(line_prev.split(',')[0])
        t2 = int(line.split(',')[0])
        dtime = t2-t1
        
        if dtime > 90 or dtime <= 30:
            print "ERROR AT LINE %d, TIME INTERVAL %d - %d, delta %d" % (i, t1, t2, dtime)
        
        line_prev = line

if __name__ == '__main__':
    conn = sqlite3.connect(fname)
    c = conn.cursor()
    c.execute('SELECT occurredAt, amount FROM usage ORDER BY timeUnit ASC')
    data = c.fetchall()
    row_prev = None
    i=0
    for row in data:
        i += 1
        if row_prev == None:
            row_prev = row
            continue
            
        dtime = row[0] - row_prev[0]
        
        if dtime > 90 or dtime < 30:
            print "skip row %d, %d to %d delta %d" % (i, row_prev[0], row[0], dtime)
            row_prev = row
            continue
        
        kw = row[1]*60*60/float(dtime)
        stamp = (row[0] + row_prev[0])/2
        #print (str(stamp)+','+str(kw)+'\n')
        out.write(str(stamp)+','+str(kw)+'\n')
        row_prev = row
    out.close()
    check_output()
    
