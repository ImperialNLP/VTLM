#!/bin/bash
CKPT="$1"
DATA_SUBDIR="$2"

if [ -z $CKPT ]; then
  echo 'You need to provide a checkpoint .pth file for pretraining'
  exit 1
fi

if [ -z $DATA_SUBDIR ]; then
  echo 'You need to provide a suffix for data subdir for masking experiments.'
  exit 1
fi

shift 2

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATA_PATH=/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe/multi30k_${DATA_SUBDIR}
DUMP_PATH="."
CUR_DIR=`dirname $0`
TRAIN=`realpath ${CUR_DIR}/../train.py`
BS=${BS:-8}
NAME="${CKPT/.pth/}_${DATA_SUBDIR}_beam${BS}/"

python $TRAIN --beam_size $BS --exp_name $NAME --exp_id "${TEST_SET}" --dump_path $DUMP_PATH \
  --reload_model "${CKPT},${CKPT}" --data_path "${DATA_PATH}" \
  --encoder_only false --lgs 'en-de' --mt_step "en-de" --emb_dim 512 --n_layers 6 --n_heads 8 \
  --dropout '0.2' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --optimizer "adam,lr=0.0001" \
  --eval_bleu true --eval_only true $@

NAME="${CKPT/.pth/}_${DATA_SUBDIR}_zeroed_beam${BS}/"
python $TRAIN --beam_size $BS --exp_name $NAME --exp_id "${TEST_SET}" --dump_path $DUMP_PATH \
  --reload_model "${CKPT},${CKPT}" --data_path "${DATA_PATH}" \
  --encoder_only false --lgs 'en-de' --mt_step "en-de" --emb_dim 512 --n_layers 6 --n_heads 8 \
  --dropout '0.2' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --optimizer "adam,lr=0.0001" \
  --eval_bleu true --eval_only true --zero_mask_emb true $@
