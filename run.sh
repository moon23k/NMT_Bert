#!/bin/bash
python3 -m pip install -U PyYAML

while getopts a:b: flag; do
    case "${flag}" in
        a) action=${OPTARG};;
        b) bert=${OPTARG};;
    esac
done

schedulers=(None cosine_annealing_warm cosine_annealing exponential step)

if [ $action='train' ]; then
    for s in "${schedulers[@]}"; do
        echo ""
        echo "$action | model: $model / bert: $bert"
        python3 train.py -model $model -bert $bert
        echo ""
    done


elif [ $action='test' ]; then
    python3 test.py -model -data -tok -sche


elif [ $action='inference' ]; then
    python3 inference.py -model -data -tok -sche
fi