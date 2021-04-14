#!/bin/bash

if [ -z $1 ]; then
  echo 'Please provide a split name (train, valid, test) for downloading.'
  exit 1
fi

split=$1
tsv_file="conceptual_captions/cc-en-de.tsv.${split}"
out_dir="conceptual_captions/images/${split}"
mkdir -p ${out_dir}

export WGETRC="`dirname $0`/wgetrc"

# Attempt to download the files
parallel --retries 2 --timeout 7 --bar -a $tsv_file --colsep '\t' wget --timeout 5 -q -O "${out_dir}/{1}" "{2}"

# Remove empty ones
find ${out_dir} -empty | xargs rm -f
