#!/bin/bash

prefix=concap_en_de_with_images.tsv
pair="de-en"

mkdir -p parallel.raw

for split in train valid test; do
  cat ${prefix}.${split} | unpaste parallel.raw/${split}.imgs parallel.raw/${split}.urls \
    parallel.raw/${split}.${pair}.en parallel.raw/${split}.${pair}.de &
done
wait
