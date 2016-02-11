2/11/2016									Atman Fozdar


EnOcean and Z-wave sensors combined setup.

Following sensors were used :


1	Door/Window contact	- EnOcean	- STM 320U
2	Switch	EnOcean	- PTM 210U
3	Temperature	- EnOcean	- STM 332U
4	Power	- Z-wave - Aeonlabs Aeotec smart energy switch
5	Light, Humidity	- Z-wave	- AeonLabs Aeotec Multisensor


Data is collected from above sensors and stored as csv in following format.


1	Date	mm/dd/yyyy
2	Time	hh:mm:ss
3	Power (Z-wave)	Watts
4	Door/Window contact sensor (EnOcean)	0 = Closed, 1 = Open
5	Temperature (Z-wave)	Deg Celcius
6	Switch (EnOcean)	0 = Pressed, 1 = Released
7	Light (Z-wave)	Luminescence
7	Humidity (Z-wave)	Percentage


Sensors were set up in Atman’s residence, replicating real home usage. Setup consisted of two Raspberry Pi’s running FHEM (for EnOcean) and Z-way server (for Z-wave) respectively.
Data from all the sensors were collected over a period of 24 hours at 1 minute interval.
read_sensors.py , is a python script to query sensors and to store it as csv.

User defined interval (in seconds) has to be specified.

In terminal : ‘python read_sensors.py 60’ for 1 minute interval.
