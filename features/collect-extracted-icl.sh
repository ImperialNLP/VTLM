#!/bin/bash

# 0-based files
find /vol/bitbucket/ocaglaya/cc_feats/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12_ghconfig_reexport -type f | \
  sed 's/.*\/\([0-9]*\).pbz2/\1/' | sort -n > icl.extracted
bzip2 -f icl.extracted
