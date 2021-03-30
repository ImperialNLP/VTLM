#!/bin/bash

DATA_PATH="/data/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe/multi30k"
DUMP_PATH="/data/ozan/experiments/mmvc/mmvc_code"
FEAT_PATH="/data/ozan/datasets/multi30k/features/oidv4/avgpool"

python train.py --exp_name vtlm_en_de_m30k_36regs --dump_path $DUMP_PATH --data_path $DATA_PATH \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' --gelu_activation true --batch_size 64 --bptt 1 \
  --optimizer 'adam,lr=0.0001' --epoch_size 29000 --max_epoch 100000 \
  --validation_metrics '_valid_en_de_mlm_ppl' --stopping_criterion '_valid_en_de_mlm_ppl,50' \
  --fp16 false --save_periodic 5 \
  --image_names $DATA_PATH --region_feats_path $FEAT_PATH --visual_first true \
  --num_of_regions 36 --only_vlm true --eval_vlm true --iter_seed 12345 --other_seed 12345 $@
