#!/bin/bash
DATA_PATH=/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe/multi30k
DUMP_PATH=/data/ozan/experiments/mmvc/mmvc_code/multi30k_mmt/from_vtlm_decinit
FEAT_PATH=/data/ozan/datasets/multi30k/features/oidv4/avgpool

CUR_DIR=`dirname $0`
TRAIN=`realpath ${CUR_DIR}/../train.py`

CKPT="$1"

if [ -z $CKPT ]; then
  echo 'You need to provide a checkpoint .pth file for pretraining'
  exit 1
fi

shift 1

# sth like periodic-xxx.pth or best-...pth
CKPT_NAME=`basename $CKPT | tr -- '-.' '_'`
CKPT_ID=$(basename `dirname $CKPT`)
CKPT_NAME="${CKPT_ID}_${CKPT_NAME}"

LOG="`dirname $CKPT`/train.log"
# Fetch previous args
PREV_ARGS=`egrep '(emb_dim|n_layers|n_heads):' $LOG | sed 's#\s*\([a-z_]*\): \([0-9]*\)$#--\1 \2#'`

PAIR=$(basename `ls ${DATA_PATH}/train.*pth | head -n1` | cut -d'.' -f2)
L1=`echo $PAIR | cut -d'-' -f1`
EPOCH=`wc -l ${DATA_PATH}/train.${PAIR}.$L1 | head -n1 | cut -d' ' -f1`
BS=${BS:-64}
LR=${LR:-0.00001}
NAME="${CKPT_NAME}_ftune_bs${BS}_lr${LR}_4regs"
PREFIX=${PREFIX:-}
DUMP_PATH="${DUMP_PATH}/${PREFIX}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python $TRAIN --beam_size 1 --exp_name ${NAME} --dump_path ${DUMP_PATH} \
  --reload_model "${CKPT},${CKPT}" --data_path ${DATA_PATH} --encoder_only false \
  --lgs 'en-de' --mmt_step "en-de" $PREV_ARGS \
  --dropout '0.2' --attention_dropout '0.1' --gelu_activation true \
  --batch_size ${BS} --optimizer "adam,lr=${LR}" \
  --epoch_size ${EPOCH} --eval_bleu true --max_epoch 500 \
  --stopping_criterion 'valid_en-de_mmt_bleu,20' --validation_metrics 'valid_en-de_mmt_bleu' \
  --region_feats_path $FEAT_PATH --image_names ${DATA_PATH} --visual_first true \
  --num_of_regions 4 --init_dec_from_enc $@
