#!/bin/bash

FEAT_PATH="/data2/ozan/conceptual_captions/avgpool_features"
DATA_PATH="/data2/ozan/conceptual_captions/mmvc_icl_data/mono.tok.bpe"
DUMP_PATH="/data/ozan/experiments/mmvc/mmvc_code"

ipython -i train.py -- --exp_name vmlm_de_v2 --dump_path $DUMP_PATH \
  --data_path $DATA_PATH \
  --lgs en-de --clm_steps '' --mlm_steps de --emb_dim 512 \
  --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' \
  --gelu_activation true --batch_size 64 --bptt 1 \
  --optimizer 'adam,lr=0.0001' --epoch_size 300000 --max_epoch 100000 \
  --validation_metrics _valid_mlm_ppl \
  --stopping_criterion '_valid_mlm_ppl,50' --fp16 false --save_periodic 5 \
  --image_names $DATA_PATH --region_feats_path $FEAT_PATH \
  --only_vlm true --load_vlm_mono true --eval_vlm true $@
