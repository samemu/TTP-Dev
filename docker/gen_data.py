import random
from sys import argv

out = open('./data.csv', 'w')
out.write('')
out.flush()

cols = int(argv[1])

for i in range(60000):
	for i in range(cols-1):
		out.write(str(random.random()) + ",")
		out.flush()
	out.write(str(random.random()) + '\n')
	out.flush()
out.close()
