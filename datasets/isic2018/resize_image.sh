#!/bin/bash

for im in ./$1/*
do 
	echo $im
	b=$(basename $im .jpg)
	echo $b
	convert $im -resize 25% ./$2/${b}.jpg
done


