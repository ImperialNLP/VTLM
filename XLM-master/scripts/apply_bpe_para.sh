#!/bin/bash
pair=en-de
OUTPATH=data/processed/XLM_en_de/6k
FASTBPE=tools/fastBPE/fast
mkdir -p $OUTPATH

for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    $FASTBPE applybpe $OUTPATH/$split.de-en.$lg data/conceptual_captions/$pair.$lg.$split   $OUTPATH/codes
  done
done
cat $OUTPATH/train.{en,de} | $FASTBPE getvocab - > $OUTPATH/vocab

# remove old binarized files
rm -rf ${OUTPATH}/*.pth

for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    python3 preprocess.py $OUTPATH/vocab $OUTPATH/$split.de-en.$lg &
  done
done
