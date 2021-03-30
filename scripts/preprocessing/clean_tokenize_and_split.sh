#!/bin/bash

pair="en-de"

# data paths
PARA_PATH=$PWD/data/conceptual_captions

# Point $MOSES to a mosesdecoder checkout
MOSES=
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl

if [ -z $MOSES ]; then
  echo "Please set MOSES to a mosesdecoder checkout folder in $0"
  exit 1
fi


# tools paths
TOKENIZE=scripts/tokenize.sh
LOWER_REMOVE_ACCENT=scripts/lowercase_and_remove_accent.py

# tokenize
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  if [ ! -f $PARA_PATH/$pair.$lg.all ]; then
  cat - | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape -threads $N_THREADS -l $lg

    cat $PARA_PATH/*.$pair.$lg | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.all
  fi
done

# split into train / valid / test
split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NTRAIN=$((NLINES - 10000));
    NVAL=$((NTRAIN + 5000));
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN             > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NVAL | tail -5000  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -5000                > $4;
}
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  split_data $PARA_PATH/$pair.$lg.all $PARA_PATH/$pair.$lg.train $PARA_PATH/$pair.$lg.valid $PARA_PATH/$pair.$lg.test
done
