#!/bin/bash

for i in {0..10}
do
    frac=$(echo "scale=1; $i / 10" | bc -l)
    echo $frac
    python cnn-train-unbalanced.py 1 $frac 1
done
