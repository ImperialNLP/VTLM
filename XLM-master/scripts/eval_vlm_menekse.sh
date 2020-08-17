#!/bin/bash
FEAT_PATH="/data2/ozan/conceptual_captions/avgpool_features"
DATA_PATH="/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe"
DUMP_PATH="/data/ozan/experiments/mmvc/mmvc_code/xlm_en_de_img/1t2emrzp5m"
MODEL="${DUMP_PATH}/periodic-${1}.pth"

shift 1

python train.py --exp_name xlm_en_de_img --dump_path "${DUMP_PATH}_probing" \
  --data_path $DATA_PATH --reload_model $MODEL \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
  --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' \
  --gelu_activation true --batch_size 64 --bptt 256 \
  --image_names $DATA_PATH --only_vlm True \
  --region_feats_path $FEAT_PATH \
  --fp16 false --eval_only true $@
