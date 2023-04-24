#!/bin/sh
python train.py --lr 0.001 --optimizer Adam --epochs 200 --deterministic --compress policies/schedule.yaml --model ai85netextrasmall --dataset MNIST --confusion --param-hist --pr-curves --embedding --device MAX78000 "$@"
