#!/usr/bin/env python
import pickle as pkl
import bz2
import sys

"""Dumps list of detected OID labels for the requested bzipped pickle files."""


if __name__ == '__main__':
    # Load labels
    with open('metadata/oid_labels.pkl', 'rb') as f:
        # 0-index labels [0, 600]
        labelmap = pkl.load(f)

    for pbz2 in sys.argv[1:]:
        with bz2.BZ2File(pbz2, 'rb') as f:
            dets = set(pkl.load(f)['detection_classes'])
            lbls = [labelmap[lbl] for lbl in dets]
            fname = pbz2.split('/')[-1].split('.')[0]
            print(f'{fname:<10} {",".join(lbls)}')
