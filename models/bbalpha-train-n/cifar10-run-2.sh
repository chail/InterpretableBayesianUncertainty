#!/bin/bash

for i in {1..10}
do
    frac=$(echo "scale=1; $i / 10" | bc -l)
    echo $frac
    python cnn-train-frac.py cifar10 0.5 2 $frac
done
