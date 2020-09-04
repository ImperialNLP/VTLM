#!/bin/bash

# receive a checkpoint
CKPT=$1

if [ -z $CKPT ]; then
  echo "You need to give a checkpoint."
  exit
fi
shift 1

DATA_PATH="/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe.nodot"
DUMP_PATH=${CKPT/.pth/_probes_nodot/}
TRAIN=`dirname $0`/../train.py

python $TRAIN --exp_name tlm_en_de_vanilla --dump_path "${DUMP_PATH}" \
  --data_path $DATA_PATH --reload_model $CKPT \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
  --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' \
  --gelu_activation true --batch_size 64 --bptt 256 \
  --fp16 false --eval_only true $@

python $TRAIN --exp_name tlm_en_de_drop_en --dump_path "${DUMP_PATH}" \
  --data_path $DATA_PATH --reload_model $CKPT \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
  --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' \
  --gelu_activation true --batch_size 64 --bptt 256 \
  --fp16 false --eval_only true --word_pred 0 --eval_probes drop_last:en $@

python $TRAIN --exp_name tlm_en_de_drop_de --dump_path "${DUMP_PATH}" \
  --data_path $DATA_PATH --reload_model $CKPT \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
  --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' \
  --gelu_activation true --batch_size 64 --bptt 256 \
  --fp16 false --eval_only true --word_pred 0 --eval_probes drop_last:de $@

python $TRAIN --exp_name tlm_en_de_drop_both --dump_path "${DUMP_PATH}" \
  --data_path $DATA_PATH --reload_model $CKPT \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
  --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' \
  --gelu_activation true --batch_size 64 --bptt 256 \
  --fp16 false --eval_only true --word_pred 0 --eval_probes "drop_last:en-de" $@
