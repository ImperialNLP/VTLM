#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python tf2_extractor.py -o /data/ozan/datasets/multi30k/features/oidv4/train \
-l /data/ozan/datasets/multi30k/images/train/index.txt -i /data/ozan/datasets/multi30k/images/train -f picklebz2 -p &

for sp in val test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr; do
  CUDA_VISIBLE_DEVICES=1 python tf2_extractor.py -o /data/ozan/datasets/multi30k/features/oidv4/${sp} \
  -l /data/ozan/datasets/multi30k/images/${sp}/index.txt -i /data/ozan/datasets/multi30k/images/${sp} -f picklebz2 -p
done
