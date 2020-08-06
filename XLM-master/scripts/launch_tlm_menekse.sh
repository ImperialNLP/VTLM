#!/bin/bash
DATA_PATH="/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe"
DUMP_PATH="/data/ozan/experiments/mmvc/mmvc_code"

python train.py --exp_name tlm_en_de --dump_path $DUMP_PATH \
  --data_path $DATA_PATH \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
  --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' \
  --gelu_activation true --batch_size 64 --bptt 256 \
  --optimizer 'adam,lr=0.0001' --epoch_size 300000 --max_epoch 100000 \
  --validation_metrics _valid_en_de_mlm_ppl \
  --stopping_criterion '_valid_en_de_mlm_ppl,25' --fp16 false \
  --save_periodic 2
