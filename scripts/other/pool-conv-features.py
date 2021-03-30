#!/usr/bin/env python
import bz2
import sys
import pickle as pkl

from pathlib import Path

from tqdm import tqdm


if __name__ == '__main__':
    try:
        inp_folder = Path(sys.argv[1])
        out_folder = Path(sys.argv[2])
    except IndexError:
        print(f'Usage: {sys.argv[0]} <inp folder or file list> <out folder>')
        sys.exit(1)

    out_folder.mkdir(exist_ok=True, parents=True)

    if inp_folder.is_dir():
        # traverse folder
        files = inp_folder.glob('*.pbz2')
    else:
        # process a text file with list of filenames
        files = []
        with open(str(inp_folder)) as f:
            for line in f:
                files.append(Path(line.strip()))

    for fname in tqdm(files):
        try:
            with bz2.BZ2File(fname, 'rb') as f:
                data = pkl.load(f)
        except Exception:
            print(f'Problem reading {fname}')
            continue

        # NxHxWxC
        feats = data.pop('detection_features').reshape(data['num_detections'], -1, 1536)

        # unnecessary as it can be inferred from the feats tensor's shape
        del data['num_detections']

        # mean pool over spatial dims
        data['detection_features'] = feats.mean(1)

        # dump
        with open(out_folder / fname.name.replace('pbz2', 'pkl'), 'wb') as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)
