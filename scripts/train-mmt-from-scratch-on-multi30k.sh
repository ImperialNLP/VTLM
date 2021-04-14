#!/bin/bash

# Trains an NMT from scratch on Multi30k

DATA_PATH="./data/multi30k"
FEAT_PATH="./data/multi30k/features"
DUMP_PATH="./models"
EXP_NAME="mmt-from-scratch-multi30k"
EPOCH_SIZE=29000

python train.py --beam_size 8 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} \
  --data_path ${DATA_PATH} --encoder_only false \
  --lgs 'en-de' --mmt_step "en-de" --emb_dim 512 --n_layers 6 --n_heads 8 \
  --dropout '0.4' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000" \
  --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 500 \
  --stopping_criterion 'valid_en-de_mmt_bleu,20' --validation_metrics 'valid_en-de_mmt_bleu' \
  --iter_seed 12345 --other_seed 12345 \
  --region_feats_path $FEAT_PATH --image_names $DATA_PATH \
  --visual_dropout '0.0' --visual_first true --num_of_regions 36 $@
