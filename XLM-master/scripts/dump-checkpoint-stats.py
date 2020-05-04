#!/usr/bin/env python
from collections import defaultdict

import sys
import torch


if __name__ == '__main__':
    for ckpt in sys.argv[1:]:
        x = torch.load(ckpt, map_location='cpu')
        #print(f"{ckpt} {x['best_metrics']} -- best_stopping_criterion: {x['best_stopping_criterion']}")
        if 'model' in x:
            params = x['model']
            norms = [('enc_' + n, p.norm(2).item()) for n, p in params.items()]
        else:
            enc_norms = [('enc_' + n, p.norm(2).item()) for n, p in x['encoder'].items()]
            dec_norms = [('dec_' + n, p.norm(2).item()) for n, p in x['decoder'].items()]
            norms = enc_norms + dec_norms

        avg_norms = defaultdict(list)
        for name, norm in norms:
            if name.endswith('weight'):
                avg_norms[name.split('.')[0]].append(norm)

        avg_norms = [(name, sum(norms) / len(norms)) for name, norms in avg_norms.items()]
        avg_norms = sorted(avg_norms, key=lambda x: x[1])

        for (name, norm) in avg_norms:
            print(f'{name:<40} avgnorm: {norm:10.2f}')
