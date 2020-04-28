#!/bin/bash
DATA_PATH=/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe/multi30k
DUMP_PATH=/data/ozan/experiments/xlm_mmvc_multi30k_ftune

CUR_DIR=`dirname $0`
TRAIN=`realpath ${CUR_DIR}/../train.py`

# Best pretrained checkpoint
CKPT="/data/ozan/experiments/xlm_mmvc/concap_emb512_6l_8h_bs512_lr0.0005/r0z5yu8md4/best-valid_en_de_mlm_ppl.pth"
LOG="`dirname $CKPT`/train.log"
# Fetch previous args
PREV_ARGS=`egrep '(emb_dim|n_layers|n_heads):' $LOG | sed 's#\s*\([a-z_]*\): \([0-9]*\)$#--\1 \2#'`

PAIR=$(basename `ls ${DATA_PATH}/train.*pth | head -n1` | cut -d'.' -f2)
L1=`echo $PAIR | cut -d'-' -f1`
EPOCH=`wc -l ${DATA_PATH}/train.${PAIR}.$L1 | head -n1 | cut -d' ' -f1`
BS=${BS:-64}
LR=${LR:-0.00001}
NAME="ftune_bs${BS}_lr${LR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ipython -i $TRAIN -- --beam_size 1 --exp_name ${NAME} --dump_path ${DUMP_PATH} \
  --reload_model "${CKPT},${CKPT}" --data_path ${DATA_PATH} --encoder_only false \
  --lgs 'en-de' --mt_step "en-de" $PREV_ARGS \
  --dropout '0.1' --attention_dropout '0.1' --gelu_activation true \
  --batch_size ${BS} --bptt 256 --optimizer "adam,lr=${LR}" \
  --epoch_size ${EPOCH} --eval_bleu true \
  --stopping_criterion 'valid_en-de_mt_bleu,20' --validation_metrics 'valid_en-de_mt_bleu' $@
