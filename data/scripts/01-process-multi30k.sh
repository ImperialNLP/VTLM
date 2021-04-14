#!/bin/bash

# Export moses path
PATH=mosesdecoder/scripts/tokenizer:${PATH}

REPLACE_UNICODE_PUNCT=replace-unicode-punctuation.perl
NORM_PUNC=normalize-punctuation.perl
REM_NON_PRINT_CHAR=remove-non-printing-char.perl
TOKENIZER=tokenizer.perl
LOWERCASE=lowercase.perl

ROOT=`dirname $0`
ROOT=`realpath $ROOT`
LOWER_REMOVE_ACCENT="${ROOT}/lowercase_and_remove_accent.py"
BINARIZE="${ROOT}/../../preprocess.py"


DATA_PATH="multi30k"
RAW_PATH="multi30k/raw"

if [ ! -d ${RAW_PATH} ]; then
  echo "Please run scripts/00-download.sh first"
  exit 1
fi

for lg in en de; do
  for split in train val test_2016_flickr test_2017_flickr test_2018_flickr test_2017_mscoco; do
    if [ $split == "val" ]; then
      osplit="valid"
    else
      osplit=${split}
    fi

    file="${RAW_PATH}/${split}.${lg}"
    if [ -f ${file}.gz ]; then
      zcat "${file}.gz" | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | \
        $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape -threads 2 -l $lg | $LOWER_REMOVE_ACCENT | \
        fastbpe applybpe_stream bpe/bpe50k.codes bpe/bpe50k.vocab > ${DATA_PATH}/${osplit}.de-en.${lg} &
    fi
  done
done
wait

# binarize
pushd ${DATA_PATH}
rm *.pth
for file in *.de-en.{en,de}; do
  python $BINARIZE ../bpe/bpe50k.vocab $file
done

# Create test links
for f in test_2016_flickr*; do
  ln -s ${f} ${f/_2016_flickr/}
done

ln -s val.order valid.order
