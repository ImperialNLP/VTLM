#!/bin/bash
CKPT="$1"

if [ -z $CKPT ]; then
  echo 'You need to provide a checkpoint .pth file for pretraining'
  exit 1
fi

shift 1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATA_PATH=/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe/multi30k
FEAT_PATH=/data/ozan/datasets/multi30k/features/oidv4/avgpool
DUMP_PATH="."

CUR_DIR=`dirname $0`
TRAIN=`realpath ${CUR_DIR}/../train.py`

BS=${BS:-1}
NAME="${CKPT/.pth/}_beam${BS}/"

for TEST_SET in test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr; do
  python $TRAIN --beam_size ${BS} --exp_name $NAME --exp_id "${TEST_SET}" --dump_path $DUMP_PATH \
    --reload_model "${CKPT},${CKPT}" --data_path "${DATA_PATH}_${TEST_SET}" \
    --encoder_only false --lgs 'en-de' --mmt_step "en-de" --emb_dim 512 --n_layers 6 --n_heads 8 \
    --dropout '0.4' --attention_dropout '0.1' --gelu_activation true \
    --batch_size 64 --optimizer "adam,lr=0.0001" \
    --eval_bleu true --eval_only true \
    --region_feats_path $FEAT_PATH --image_names ${DATA_PATH}_${TEST_SET} $@
done
