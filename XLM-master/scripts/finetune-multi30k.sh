#!/bin/bash

CUR_DIR=`dirname $0`
DATA_PATH=/data2/ozan/conceptual_captions/prep_data_50k_fixed/multi30k
TRAIN=`realpath ${CUR_DIR}/../train.py`

# Best pretrained checkpoint
CKPT_ID="6hunzkv62d"
CKPT=/data/ozan/experiments/xlm_mmvc_v100/xlm_en_de_tlm/${CKPT_ID}/best-valid_en_de_mlm_ppl.pth
LOG="`dirname $CKPT`/train.log"
EXP_NAME="multi30k_en_de_ftune_${CKPT_ID}"
DUMP_PATH=/data/ozan/experiments/xlm_mmvc/${EXP_NAME}

PREV_ARGS=`egrep '(emb_dim|n_layers|n_heads):' $LOG | sed 's#\s*\([a-z_]*\): \([0-9]*\)$#--\1 \2#'`

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ipython -i $TRAIN -- --beam_size 1 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} \
  --reload_model "${CKPT},${CKPT}" --data_path ${DATA_PATH} --encoder_only false \
  --lgs 'en-de' --mt_step "en-de" $PREV_ARGS \
  --dropout '0.1' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --bptt 256 --optimizer 'adam,lr=0.00001' \
  --epoch_size 29000 --eval_bleu true \
  --stopping_criterion 'valid_en-de_mt_bleu,20' --validation_metrics 'valid_en-de_mt_bleu' $@
