l1=0
docker ps | while read -r item
do
	if [ $l1 -eq 0 ]
	then
		l1=1
		#echo "skip"
		continue
	fi
	#echo "$item" | cut -c1-12
	id=`expr "$item" | cut -c1-12`
	#echo "$id"
	if [ $l1 -eq 1 ]
	then
		l1=2
		echo "first copy: $id"
		docker cp "$id":/usr/src/app/current.csv ./current.csv
	else
		echo "copying: $id"
		docker cp "$id":/usr/src/app/current.csv ./temp.csv
		paste -d, current.csv temp.csv > temp2.csv
		cp temp2.csv current.csv
		rm temp2.csv
	fi
done
echo "done."
