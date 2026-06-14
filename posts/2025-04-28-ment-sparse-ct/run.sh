#!/bin/bash

for n in 6 12 25 50 100
do
    echo $n
    python train.py --nmeas=$n
    mv outputs/fig_compare_all.png images/fig_compare_all_$n.png
done

