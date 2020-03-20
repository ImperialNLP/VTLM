#!/bin/bash

for GPU in 0 1 2 3 5 6 7 8; do
  CUDA_VISIBLE_DEVICES=$GPU python scripts/tf2_extractor.py -l features/remaining_gt_2M.txt.0${GPU} \
    -f picklebz2 -o /vol/bitbucket/ocaglaya/cc_feats -O 1 -p &
done
wait
