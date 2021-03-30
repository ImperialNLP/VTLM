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


DATA_PATH="conceptual_captions"
PREFIX="cc-en-de.tsv"

# Explode the TSVs
for split in train valid test; do
  echo "Exploding TSV for ${split}"
  cut -f1 < ${DATA_PATH}/${PREFIX}.${split} > ${DATA_PATH}/${split}.imgs
  cut -f2 < ${DATA_PATH}/${PREFIX}.${split} > ${DATA_PATH}/${split}.urls

  # English is already tokenized
  cut -f3 < ${DATA_PATH}/${PREFIX}.${split} \
    | $REPLACE_UNICODE_PUNCT \
    | $REM_NON_PRINT_CHAR | $LOWER_REMOVE_ACCENT > ${DATA_PATH}/${split}.en

  # Tokenize German captions
  cut -f4 < ${DATA_PATH}/${PREFIX}.${split} \
    | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l de \
    | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape -threads 8 -l de \
    | $LOWER_REMOVE_ACCENT > ${DATA_PATH}/${split}.de
done

# Apply the pre-learned BPE
for split in train valid test; do
  for lg in en de; do
    if [ ! -f "${DATA_PATH}/${split}.de-en.${lg}" ]; then
      cat ${DATA_PATH}/${split}.${lg} | fastbpe applybpe_stream bpe/bpe50k.codes bpe/bpe50k.vocab \
        > "${DATA_PATH}/${split}.de-en.${lg}" &
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
