#!/usr/bin/env python
import pickle as pkl
from pathlib import Path
import bz2
import sys

"""Dumps list of detected OID labels for the requested bzipped pickle files."""


if __name__ == '__main__':
    # Load labels
    label_fname = Path('~/git/Animal/metadata/oid_labels.pkl').expanduser()
    with open(str(label_fname), 'rb') as f:
        # 0-index labels [0, 600]
        labelmap = pkl.load(f)

    for pbz2 in sys.argv[1:]:
        fname = pbz2.split('/')[-1].split('.')[0]
        try:
            with open(pbz2, 'rb') as f:
                dets = set(pkl.load(f)['detection_classes'].flatten())
                lbls = [labelmap[lbl] for lbl in dets]
                print(f'{fname:<10} {",".join(lbls)}')
        except Exception as exc:
            print(f'{fname} buggy!')
            raise(exc)
