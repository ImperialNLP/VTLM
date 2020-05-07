#!/bin/bash
DATA_PATH=/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe/multi30k
DUMP_PATH=/data/ozan/experiments/xlm_mmvc_multi30k_scratch

NAME=$1

if [ -z $NAME ]; then
  echo 'provide a name for the experiment'
  exit 1
fi
shift 1

CUR_DIR=`dirname $0`
TRAIN=`realpath ${CUR_DIR}/../train.py`

PAIR=$(basename `ls ${DATA_PATH}/train.*pth | head -n1` | cut -d'.' -f2)
L1=`echo $PAIR | cut -d'-' -f1`
EPOCH=`wc -l ${DATA_PATH}/train.${PAIR}.$L1 | head -n1 | cut -d' ' -f1`

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python $TRAIN --beam_size 1 --exp_name ${NAME} --dump_path ${DUMP_PATH} \
  --data_path ${DATA_PATH} --encoder_only false \
  --lgs 'en-de' --mt_step "en-de" $PREV_ARGS \
  --dropout '0.1' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --optimizer "adam,lr=0.00001" \
  --epoch_size ${EPOCH} --eval_bleu true --max_epoch 50 \
  --stopping_criterion 'valid_en-de_mt_bleu,10' --validation_metrics 'valid_en-de_mt_bleu' $@
