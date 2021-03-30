#!/bin/bash
DATA_PATH=/data/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe/multi30k
DUMP_PATH=/data/ozan/experiments/mmvc/mmvc_code/multi30k_from_scratch_v5_do0.4_visualfirst
FEAT_PATH=/data/ozan/datasets/multi30k/features/oidv4/avgpool

EVAL_ONLY=
BS=1

for ((i = 1; i <= $#; i++ )); do
  if [[ ${!i} == "--eval_only" ]]; then
    ((i++))
    if [[ ${!i} == "true" ]]; then
      EVAL_ONLY=1
      BS=8
      DUMP_PATH="."
      break
    fi
  fi
done

CKPT=
if [[ $1 =~ \.pth$ ]]; then
  CKPT=$1
  shift 1
fi

CUR_DIR=`dirname $0`
TRAIN=`realpath ${CUR_DIR}/../train.py`

PAIR=$(basename `ls ${DATA_PATH}/train.*pth | head -n1` | cut -d'.' -f2)
L1=`echo $PAIR | cut -d'-' -f1`
EPOCH=`wc -l ${DATA_PATH}/train.${PAIR}.$L1 | head -n1 | cut -d' ' -f1`

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ -z $EVAL_ONLY ]; then
  # train
  python $TRAIN --beam_size ${BS} --exp_name mmt_36reg --dump_path ${DUMP_PATH} \
    --data_path ${DATA_PATH} --encoder_only false \
    --lgs 'en-de' --mmt_step "en-de" --emb_dim 512 --n_layers 6 --n_heads 8 \
    --dropout '0.4' --attention_dropout '0.1' --gelu_activation true \
    --batch_size 64 --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000" \
    --epoch_size ${EPOCH} --eval_bleu true --max_epoch 500 \
    --stopping_criterion 'valid_en-de_mmt_bleu,20' --validation_metrics 'valid_en-de_mmt_bleu' \
    --region_feats_path $FEAT_PATH --image_names ${DATA_PATH} \
    --visual_dropout '0.0' --visual_first true --num_of_regions 36 $@
else
  NAME="${CKPT/.pth/}_beam${BS}/"
  for TEST_SET in test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr; do
    echo python $TRAIN --beam_size ${BS} --exp_name $NAME --dump_path $DUMP_PATH \
      --data_path "${DATA_PATH}_${TEST_SET}" --encoder_only false \
      --lgs 'en-de' --mmt_step "en-de" --emb_dim 512 --n_layers 6 --n_heads 8 \
      --dropout '0.4' --attention_dropout '0.1' --gelu_activation true \
      --batch_size 64 --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000" \
      --eval_bleu true --max_epoch 500 \
      --stopping_criterion 'valid_en-de_mmt_bleu,20' --validation_metrics 'valid_en-de_mmt_bleu' \
      --region_feats_path $FEAT_PATH --image_names ${DATA_PATH}_${TEST_SET} \
      --reload_model "${CKPT},${CKPT}" --eval_only true --eval_bleu true --exp_id ${TEST_SET} \
      --visual_dropout '0.0' --visual_first true --num_of_regions 36 $@
  done
fi
