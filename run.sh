#!/bin/bash

while getopts a:m:d: flag; do
    case "${flag}" in
        a) action=${OPTARG};;
        m) model=${OPTARG};;
        d) data=${OPTARG};;
    esac
done

schedulers=(None cosine_annealing_warm cosine_annealing exponential step)

if [ $action='train' ]; then
    for s in "${schedulers[@]}"; do
        echo ""
        echo "$action | model: $model / data: $data / scheduler: $s"
        python3 train.py -model $model -data $data -scheduler $s
        echo ""
    done

elif [ $action='test' ]; then
    python3 test.py -model -data -tok -sche

elif [ $action='inference' ]; then
    python3 inference.py -model -data -tok -sche
fi