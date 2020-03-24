#!/bin/bash

for GPU in 0 1 2 3 5 6 7 8; do
  CUDA_VISIBLE_DEVICES=$GPU python scripts/tf2_extractor.py -l features/remaining.ab.0${GPU} \
    -f picklebz2 -o /data/ozan/datasets/conceptual_captions/features -O 1 -p &
done
wait
