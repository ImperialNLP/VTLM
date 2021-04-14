#!/bin/bash

# Trains TLM on Multi30k for quick / demonstrational purposes
# (The hyper-params are tuned for CC training)

DATA_PATH="./data/multi30k"
DUMP_PATH="./models"
EXP_NAME="tlm-on-multi30k"
EPOCH_SIZE=29000

python train.py --exp_name $EXP_NAME --dump_path $DUMP_PATH --data_path $DATA_PATH \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' --gelu_activation true --batch_size 64 --bptt 1 \
  --optimizer 'adam,lr=0.0001' --epoch_size ${EPOCH_SIZE} --max_epoch 100000 \
  --validation_metrics '_valid_en_de_mlm_ppl' --stopping_criterion '_valid_en_de_mlm_ppl,50' \
  --fp16 false --save_periodic 5 --iter_seed 12345 --other_seed 12345 $@
