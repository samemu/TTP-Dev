import socket
UDP_IP = "172.18.1.1"
UDP_PORT = 8088

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

logfile = open('./datalog.txt', 'w')

while True:
	data, addr = (sock.recvfrom(1024))
	print (data)
	print (addr[0])
	logfile.write( addr[0] + " " + data + "\n")
