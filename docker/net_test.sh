#!/bin/bash
ncols="$(head -1 data.csv | sed 's/[^,]//g' | wc -c)"

idx=1

echo $ncols

until [ $idx -gt $ncols ]
do
	echo $idx
	docker run --net smartnet -d rkalvait/blr
	idx=`expr $idx + 1`
done

docker run --net smartnet --ip 172.18.1.1 -d rkalvait/blr_host
