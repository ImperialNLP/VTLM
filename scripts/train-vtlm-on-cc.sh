#!/bin/bash

# Trains VTLM on Conceptual Captions

DATA_PATH="./data/conceptual_captions"
FEAT_PATH="./data/conceptual_captions/features"
DUMP_PATH="./models"
EXP_NAME="vtlm-on-cc"
EPOCH_SIZE=300000

# --region_mask_type mask: Replaces masked region feature vectors with [MASK] embedding
# --region_mask_type zero: Replaces masked region feature vectors with a 0-vector


python train.py --exp_name $EXP_NAME --dump_path $DUMP_PATH --data_path $DATA_PATH \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' --gelu_activation true --batch_size 64 --bptt 1 \
  --optimizer 'adam,lr=0.0001' --epoch_size ${EPOCH_SIZE} --max_epoch 100000 \
  --validation_metrics '_valid_en_de_mlm_ppl' --stopping_criterion '_valid_en_de_mlm_ppl,50' \
  --fp16 false --save_periodic 5 --iter_seed 12345 --other_seed 12345 \
  --image_names $DATA_PATH --region_feats_path $FEAT_PATH --visual_first true \
  --num_of_regions 36 --only_vlm true --eval_vlm true --region_mask_type mask $@
