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
TOKENIZE=scripts/tokenize.sh
LOWER_REMOVE_ACCENT=scripts/lowercase_and_remove_accent.py

# tokenize
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  if [ ! -f $PARA_PATH/$pair.$lg.all ]; then
  cat - | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape -threads $N_THREADS -l $lg

    cat $PARA_PATH/*.$pair.$lg | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.all
  fi
done
