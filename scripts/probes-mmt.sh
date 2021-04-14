#!/bin/bash
CKPT="$1"

if [ -z $CKPT ]; then
  echo 'You need to provide an NMT checkpoint .pth file for probing experiments.'
  exit 1
fi

shift 1


DUMP_PATH="."
BS=${BS:-8}

DATA_SUBDIR="mask_oid"
FEAT_PATH="./data/multi30k/features"
DATA_PATH="./data/multi30k/${DATA_SUBDIR}"

if [ ! -d $DATA_PATH ]; then
  echo "You need to prepare masked versions of sentences and re-prepare the data folder."
  exit 1
fi

NAME="${CKPT/.pth/}_${DATA_SUBDIR}_beam${BS}/"
python train.py --beam_size $BS --exp_name $NAME --exp_id "${TEST_SET}" --dump_path $DUMP_PATH \
  --reload_model "${CKPT},${CKPT}" --data_path "${DATA_PATH}" \
  --encoder_only false --lgs 'en-de' --mmt_step "en-de" --emb_dim 512 --n_layers 6 --n_heads 8 \
  --dropout '0.2' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --optimizer "adam,lr=0.0001" \
  --eval_bleu true --eval_only true --zero_mask_emb false \
  --region_feats_path $FEAT_PATH --image_names ${DATA_PATH} --visual_first true --reg_enc_bias false $@

NAME="${CKPT/.pth/}_${DATA_SUBDIR}_zero_emb_beam${BS}/"
python train.py --beam_size $BS --exp_name $NAME --exp_id "${TEST_SET}" --dump_path $DUMP_PATH \
  --reload_model "${CKPT},${CKPT}" --data_path "${DATA_PATH}" \
  --encoder_only false --lgs 'en-de' --mmt_step "en-de" --emb_dim 512 --n_layers 6 --n_heads 8 \
  --dropout '0.2' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --optimizer "adam,lr=0.0001" \
  --eval_bleu true --eval_only true --zero_mask_emb true \
  --region_feats_path $FEAT_PATH --image_names ${DATA_PATH} --visual_first true --reg_enc_bias false $@

DATA_SUBDIR="mask_removed"
NAME="${CKPT/.pth/}_${DATA_SUBDIR}_beam${BS}/"
DATA_PATH="./data/multi30k/${DATA_SUBDIR}"
if [ ! -d $DATA_PATH ]; then
  echo "You need to remove the masked Flickr30k entities from sentences and re-prepare the data folder."
  exit 1
fi

python train.py --beam_size $BS --exp_name $NAME --exp_id "${TEST_SET}" --dump_path $DUMP_PATH \
  --reload_model "${CKPT},${CKPT}" --data_path "${DATA_PATH}" \
  --encoder_only false --lgs 'en-de' --mmt_step "en-de" --emb_dim 512 --n_layers 6 --n_heads 8 \
  --dropout '0.2' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 64 --optimizer "adam,lr=0.0001" \
  --eval_bleu true --eval_only true --zero_mask_emb false \
  --region_feats_path $FEAT_PATH --image_names ${DATA_PATH} --visual_first true --reg_enc_bias false $@
