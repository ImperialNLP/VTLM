#!/bin/bash

CC_URL="https://zenodo.org/record/4646961/files/conceptual_captions_en_de.tar.bz2"
M30K_FEATS_URL="https://zenodo.org/record/4646961/files/multi30k_oidv4_features.tar.xz"
TF_OBJ_URL="https://zenodo.org/record/4646961/files/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.bz2"

if [ ! -d "faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12_ghconfig_reexport" ]; then
  if [ ! -f `basename $TF_OBJ_URL` ]; then
    wget $TF_OBJ_URL
  fi
  tar xvf `basename $TF_OBJ_URL`
fi

if [ ! -f "conceptual_captions/cc-en-de.tsv.train" ]; then
  if [ ! -f `basename $CC_URL` ]; then
    wget $CC_URL
  fi
  tar xvf `basename $CC_URL`
fi

if [ ! -d "multi30k/features" ]; then
  if [ ! -f `basename $M30K_FEATS_URL` ]; then
    wget $M30K_FEATS_URL
  fi
  tar xvf `basename $M30K_FEATS_URL` -C multi30k --strip-components=1
fi

# Checkout Multi30k corpus
if [ ! -d "multi30k/raw" ]; then
  wget https://github.com/multi30k/dataset/archive/refs/heads/master.tar.gz
  tar xvf master.tar.gz -C multi30k --wildcards '*task1/raw/*.en.gz' --strip-components=3
  tar xvf master.tar.gz -C multi30k --wildcards '*task1/raw/*.de.gz' --strip-components=3
fi

# Checkout Mosesdecoder
if [ ! -d mosesdecoder ]; then
  git clone https://github.com/moses-smt/mosesdecoder --depth 1
fi
