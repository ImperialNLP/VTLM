#!/usr/bin/env python
from pathlib import Path
from collections import defaultdict

import sys

import numpy as np
import torch


if __name__ == '__main__':
    exclude = ('projector', 'regional', 'img_pred')
    norms = defaultdict(list)
    total_norms = []

    mode = None
    if not sys.argv[1].endswith('.pth'):
        mode = sys.argv.pop(1)

    ckpts = [Path(fname) for fname in sys.argv[1:]]
    # sort by epoch order
    ckpts = sorted(ckpts, key=lambda x: int(x.name.replace('.pth', '')[9:]))

    for ckpt in ckpts:
        period = int(ckpt.name.replace('.pth', '')[9:])
        x = torch.load(ckpt, map_location='cpu')
        total_sum_square = 0
        params = {n: p for n, p in x['model'].items() if not n.startswith(exclude)}
        for name, param in params.items():
            norms[name].append((param**2).sum().sqrt().item())
            total_sum_square += (param**2).sum()
        total_norms.append(total_sum_square.sqrt().item())

    stats = {}
    for name in norms:
        vals = np.array(norms[name])
        mean = vals.mean()
        stdev = vals.std()
        final_to_init = vals[-1] / vals[0]
        stats[name] = (final_to_init, mean, stdev)

    if mode:
        if mode in norms:
            # evolution of norms for a single param requested
            vals = norms[mode]
            for i, val in enumerate(vals):
                print(i * 2, val)
    else:
        stats = sorted(stats.items(), key=lambda x: -x[1][0])
        for name, (final_to_init, mean, stdev) in stats:
            print(f'{name:<40} {mean:8.2f} {stdev:8.2f} (ratio: {final_to_init:2.2f})')

#         else:
            # enc_norms = [('enc_' + n, p.norm(2).item()) for n, p in x['encoder'].items()]
            # dec_norms = [('dec_' + n, p.norm(2).item()) for n, p in x['decoder'].items()]
            # total_sum_square += sum([(p**2).sum() for p in x['encoder'].values()])
            # total_sum_square += sum([(p**2).sum() for p in x['decoder'].values()])
            # norms = enc_norms + dec_norms

        # print(ckpt.split('/')[-1], f'{total_sum_square.sqrt().item():.2f}')

#         avg_norms = defaultdict(list)
        # for name, norm in norms:
            # if name.endswith('weight'):
                # avg_norms[name.split('.')[0]].append(norm)

        # avg_norms = [(name, sum(norms) / len(norms)) for name, norms in avg_norms.items()]
        # avg_norms = sorted(avg_norms, key=lambda x: x[1])

        # for (name, norm) in avg_norms:
            # print(f'{name:<40} avgnorm: {norm:10.2f}')
