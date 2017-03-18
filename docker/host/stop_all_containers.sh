l1=0
docker ps | while read -r item
do
	if [ $l1 -eq 0 ]
	then
		l1=1
		continue
	fi
	id=`expr "$item" | cut -c1-12`
	l1=2
	echo "stopping: $id"
	docker stop "$id"
done

echo "Done."
