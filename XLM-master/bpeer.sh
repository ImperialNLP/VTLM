#!/bin/bash
pair=en-de
OUTPATH=data/processed/XLM_en_de/30k
FASTBPE=tools/fastBPE/fast
mkdir -p $OUTPATH
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    $FASTBPE applybpe $OUTPATH/$pair.$lg.$split data/conceptual_captions/$pair.$lg.$split $OUTPATH/codes
    python preprocess.py $OUTPATH/vocab.txt $OUTPATH/$pair.$lg.$split
  done
done
