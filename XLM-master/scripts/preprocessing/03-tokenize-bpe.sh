#!/bin/bash
REPLACE_UNICODE_PUNCT=replace-unicode-punctuation.perl
NORM_PUNC=normalize-punctuation.perl
REM_NON_PRINT_CHAR=remove-non-printing-char.perl
TOKENIZER=tokenizer.perl
LOWERCASE=lowercase.perl
LOWER_REMOVE_ACCENT=${HOME}/git/Animal/XLM-master/tools/lowercase_and_remove_accent.py
BINARIZE=${HOME}/git/Animal/XLM-master/preprocess.py


inp_dir=parallel.raw
tok_dir=parallel.tok
bpe_dir=${tok_dir}.bpe
pair="de-en"

mkdir -p ${tok_dir} ${bpe_dir}

# English is already tokenized in the vanilla dataset
#for split in train valid test; do
    #file="${inp_dir}/${split}.${pair}.en"
    #cat $file | $REPLACE_UNICODE_PUNCT | \
      #$REM_NON_PRINT_CHAR | \
      #$LOWER_REMOVE_ACCENT > ${tok_dir}/${split}.${pair}.en &
#done

## Process detokenized German as usual
#for split in train valid test; do
    #file="${inp_dir}/${split}.${pair}.de"
    #cat $file | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l de | \
      #$REM_NON_PRINT_CHAR | $TOKENIZER -no-escape -threads 8 -l de | \
      #$LOWER_REMOVE_ACCENT > ${tok_dir}/${split}.${pair}.de &
#done
#wait

# Learn BPE on concatenated/unique EN-DE training pairs
echo 'concatenating training files and making it unique for BPE'
cat ${tok_dir}/train.${pair}.{en,de} | sort -u > /tmp/bpe.data
echo 'learning BPE'
fastbpe learnbpe 50000 /tmp/bpe.data > ${bpe_dir}/codes
rm /tmp/bpe.data

# apply BPE to training set
for lg in en de; do
  cat ${tok_dir}/train.${pair}.${lg} | fastbpe applybpe_stream ${bpe_dir}/codes \
    > ${bpe_dir}/train.${pair}.${lg} &
done
wait

echo 'extracting vocabulary'
fastbpe getvocab ${bpe_dir}/train.${pair}.en ${bpe_dir}/train.${pair}.de > ${bpe_dir}/vocab

# apply BPE on remaining sets now
for lg in en de; do
  for split in valid test; do
    cat ${tok_dir}/${split}.${pair}.${lg} | fastbpe applybpe_stream ${bpe_dir}/codes ${bpe_dir}/vocab \
      > ${bpe_dir}/${split}.${pair}.${lg} &
  done
done
wait

# binarize
pushd $bpe_dir
rm -rf *.pth
for file in *.{en,de}; do
  python $BINARIZE vocab $file &
done
wait
