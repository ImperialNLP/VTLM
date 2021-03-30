#!/bin/bash

# Set these first!
OUT_PATH=
MULTI30K_IMGS=

if [ -z $OUT_PATH ]; then
  echo 'Set OUT_PATH'
  exit 1
fi

if [ -z $MULTI30K_IMGS ]; then
  echo 'Set MULTI30K_IMGS'
  exit 1
fi

EXTRACTOR="`dirname $0`/tf-obj-extractor.py"

CUDA_VISIBLE_DEVICES=0 python $EXTRACTOR -o ${OUT_PATH}/train \
  -l ${MULTI30K_IMGS}/train/index.txt \
  -i ${MULTI30K_IMGS}/train -f pickle -p -P &

for sp in val test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr; do
  CUDA_VISIBLE_DEVICES=1 python $EXTRACTOR -o ${OUT_PATH}/${sp} \
  -l ${MULTI30K_IMGS}/${sp}/index.txt \
  -i ${MULTI30K_IMGS}/${sp} -f pickle -p -P
done
