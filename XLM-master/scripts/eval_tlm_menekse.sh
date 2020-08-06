#!/bin/bash
DATA_PATH="/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe"
DUMP_PATH="/data/ozan/experiments/mmvc/mmvc_code/tlm_en_de/8skyrwez71"
MODEL="${DUMP_PATH}/periodic-${1}.pth"

shift 1

ipython -i train.py -- --exp_name tlm_en_de --dump_path "${DUMP_PATH}_probing" \
  --data_path $DATA_PATH --reload_model $MODEL \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
  --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' \
  --gelu_activation true --batch_size 64 --bptt 256 \
  --fp16 false --eval_only true $@
