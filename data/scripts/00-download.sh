#!/bin/bash

CC_URL="https://zenodo.org/record/4646961/files/conceptual_captions_en_de.tar.bz2"
M30K_FEATS_URL="https://zenodo.org/record/4646961/files/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.bz2"
TF_OBJ_MODEL_URL="https://zenodo.org/record/4646961/files/multi30k_oidv4_features.tar.xz"

if [ ! -f "conceptual_captions/cc-en-de.tsv.train" ]; then
  if [ ! -f `basename $CC_URL` ]; then
    # Download
    wget $CC_URL
  fi
  # Extract
  tar xvf `basename $CC_URL`
fi
