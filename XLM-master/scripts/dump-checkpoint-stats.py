#!/usr/bin/env python
import sys
import torch


if __name__ == '__main__':
    for ckpt in sys.argv[1:]:
        x = torch.load(ckpt)
        print(f"{ckpt} {x['best_metrics']} -- best_stopping_criterion: {x['best_stopping_criterion']}")
