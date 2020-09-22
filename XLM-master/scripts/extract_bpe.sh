

OUTPATH=data/processed/XLM_en_de/8k
mkdir -p $OUTPATH
shuf -r -n 30000 data/conceptual_captions/en-de.en.train >> $OUTPATH/bpe.train
shuf -r -n 30000 data/conceptual_captions/en-de.de.train >> $OUTPATH/bpe.train

OUTPATH=data/processed/XLM_en_de/8k  # path where processed files will be stored
FASTBPE=fastBPE/fast  # path to the fastBPE tool

# create output path
mkdir -p $OUTPATH

# learn bpe codes on the training set (or only use a subset of it)
$FASTBPE learnbpe 8000 $OUTPATH/bpe.train  > $OUTPATH/codes
