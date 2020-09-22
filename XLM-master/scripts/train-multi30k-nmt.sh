#!/bin/bash
DATA_PATH=/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe/multi30k
DUMP_PATH=/data/ozan/experiments/mmvc/mmvc_code/multi30k_from_scratch_nmt_v5

CUR_DIR=`dirname $0`
TRAIN=`realpath ${CUR_DIR}/../train.py`

PAIR=$(basename `ls ${DATA_PATH}/train.*pth | head -n1` | cut -d'.' -f2)
L1=`echo $PAIR | cut -d'-' -f1`
EPOCH=`wc -l ${DATA_PATH}/train.${PAIR}.$L1 | head -n1 | cut -d' ' -f1`

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python $TRAIN --beam_size 1 --exp_name nmt_fixed_seed --dump_path ${DUMP_PATH} \
  --data_path ${DATA_PATH} --encoder_only false \
  --lgs 'en-de' --mt_step "en-de" --emb_dim 512 --n_layers 6 --n_heads 8 \
  --dropout '0.4' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000" \
  --epoch_size ${EPOCH} --eval_bleu true --max_epoch 500 \
  --stopping_criterion 'valid_en-de_mt_bleu,20' --validation_metrics 'valid_en-de_mt_bleu' $@
