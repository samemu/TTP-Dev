#!/bin/bash
ncols="$(head -1 data.csv | sed 's/[^,]//g' | wc -c)"

idx=0

echo $ncols
docker run --net smartnet --ip 172.18.1.1 -d markzz/blr_host
#docker run --net smartnet --ip 172.18.1.1 -d rkalvait/blr_host #uncomment for testing

until [ $idx -eq $ncols ]
do
	echo $idx
	docker run --net smartnet -d markzz/blr python report_hourly.py $idx
	#docker run --net smartnet -d rkalvait/blr python report_hourly.py $idx #uncomment for testing
	idx=`expr $idx + 1`
done

