#!/bin/bash

# receive a checkpoint
CKPT=$1

if [ -z $CKPT ]; then
  echo "You need to give a pre-trained checkpoint."
  exit
fi
shift 1

DATA_PATH="./data/conceptual_captions/dots_removed"
if [ ! -d $DATA_PATH ]; then
  echo "You need to re-prepare a new dataset folder where final dots are removed from valid/test set sentences"
  exit 1
fi

FEAT_PATH="./data/conceptual_captions/features"
DUMP_PATH=${CKPT/.pth/_probes_nodot/}

python train.py --exp_name vtlm_en_de_img_vanilla --dump_path "${DUMP_PATH}" \
  --data_path $DATA_PATH --reload_model $CKPT \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
  --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' \
  --gelu_activation true --batch_size 64 --bptt 256 \
  --image_names $DATA_PATH --only_vlm True \
  --region_feats_path $FEAT_PATH \
  --fp16 false --eval_only true --eval_vlm true $@

python train.py --exp_name vtlm_en_de_img_drop_en --dump_path "${DUMP_PATH}" \
  --data_path $DATA_PATH --reload_model $CKPT \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
  --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' \
  --gelu_activation true --batch_size 64 --bptt 256 \
  --image_names $DATA_PATH --only_vlm True \
  --region_feats_path $FEAT_PATH \
  --fp16 false --eval_only true --eval_vlm true --word_pred 0 --eval_probes drop_last:en $@

python train.py --exp_name vtlm_en_de_img_drop_de --dump_path "${DUMP_PATH}" \
  --data_path $DATA_PATH --reload_model $CKPT \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
  --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' \
  --gelu_activation true --batch_size 64 --bptt 256 \
  --image_names $DATA_PATH --only_vlm True \
  --region_feats_path $FEAT_PATH \
  --fp16 false --eval_only true --eval_vlm true --word_pred 0 --eval_probes drop_last:de $@

python train.py --exp_name vtlm_en_de_img_drop_both --dump_path "${DUMP_PATH}" \
  --data_path $DATA_PATH --reload_model $CKPT \
  --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
  --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' \
  --gelu_activation true --batch_size 64 --bptt 256 \
  --image_names $DATA_PATH --only_vlm True \
  --region_feats_path $FEAT_PATH \
  --fp16 false --eval_only true --eval_vlm true --word_pred 0 --eval_probes "drop_last:en-de" $@

for i in `seq 1 5`; do
  python train.py --exp_name vtlm_en_de_img_drop_en_shuf --dump_path "${DUMP_PATH}" \
    --data_path $DATA_PATH --reload_model $CKPT \
    --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
    --n_layers 6 --n_heads 8 --dropout '0.1' \
    --attention_dropout '0.1' \
    --gelu_activation true --batch_size 64 --bptt 256 \
    --image_names $DATA_PATH --only_vlm True \
    --region_feats_path $FEAT_PATH \
    --fp16 false --eval_only true --eval_vlm true --word_pred 0 --eval_probes drop_last:en --eval_image_order shuffle $@

  python train.py --exp_name vtlm_en_de_img_drop_de_shuf --dump_path "${DUMP_PATH}" \
    --data_path $DATA_PATH --reload_model $CKPT \
    --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
    --n_layers 6 --n_heads 8 --dropout '0.1' \
    --attention_dropout '0.1' \
    --gelu_activation true --batch_size 64 --bptt 256 \
    --image_names $DATA_PATH --only_vlm True \
    --region_feats_path $FEAT_PATH \
    --fp16 false --eval_only true --eval_vlm true --word_pred 0 --eval_probes drop_last:de --eval_image_order shuffle $@

  python train.py --exp_name vtlm_en_de_img_drop_both_shuf --dump_path "${DUMP_PATH}" \
    --data_path $DATA_PATH --reload_model $CKPT \
    --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 \
    --n_layers 6 --n_heads 8 --dropout '0.1' \
    --attention_dropout '0.1' \
    --gelu_activation true --batch_size 64 --bptt 256 \
    --image_names $DATA_PATH --only_vlm True \
    --region_feats_path $FEAT_PATH \
    --fp16 false --eval_only true --eval_vlm true --word_pred 0 --eval_probes "drop_last:en-de" --eval_image_order shuffle $@
done
