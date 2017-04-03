import random
from sys import argv

if __name__ == '__main__':
	cols = int(argv[1])
	out = open('./data.csv', 'w')
	for i in range(60000):
		for i in range(cols-1):
			out.write(str(random.random()) + ",")
		out.write(str(random.random()) + '\n')
	out.close()
