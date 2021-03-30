#!/usr/bin/env python

"""This script requires TensorFlow 2.x and a SavedModel export of obj-det
models."""

import os
import sys
import time
import uuid
from pathlib import Path
from multiprocessing import Pool
import argparse
import bz2
import gzip
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable thread explosion
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

import numpy as np
import tqdm


OUT_FIELDS = [
    'detection_scores',
    'detection_classes',
    'detection_boxes',
    'detection_features',
]


def prepare_dict(output, class_offset=0, num_proposals=10000, pool=False):
    d = {k: output[k].numpy().squeeze(0)[:num_proposals] for k in OUT_FIELDS}
    d['num_detections'] = min(int(output['num_detections']), num_proposals)
    d['detection_classes'] = (d['detection_classes'] - float(class_offset)).astype(np.uint16)

    if pool:
        feats = d.pop('detection_features').reshape(d['num_detections'], -1, 1536)

        # mean pool over spatial dims
        d['detection_features'] = feats.mean(1)

    return d


def read_image_list(fname):
    fnames = []
    with open(fname) as f:
        for line in f:
            fnames.append(line.strip())
    return fnames


def fn_pickle(feat_dict, fname):
    with open(fname, 'wb') as f:
        pickle.dump(feat_dict, f, protocol=4, fix_imports=False)


def fn_picklegz(feat_dict, fname):
    with gzip.GzipFile(fname, 'wb', compresslevel=2) as f:
        pickle.dump(feat_dict, f, protocol=4, fix_imports=False)


def fn_picklebz2(feat_dict, fname):
    with bz2.BZ2File(fname, 'wb', compresslevel=1) as f:
        pickle.dump(feat_dict, f, protocol=4, fix_imports=False)


def fn_npz(feat_dict, fname):
    np.savez_compressed(fname, **feat_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='tfobj-extractor')
    def_vtlm_path = os.environ.get('VTLM_PATH', '.')

    parser.add_argument('-m', '--model-folder', type=str,
                        default=f"{def_vtlm_path}/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12_ghconfig_reexport",
                        help='Model folder where checkpoints reside.')

    parser.add_argument('-i', '--img-root', default=f'{def_vtlm_path}/images',
                        help='Image root to be prepended to every image name.')

    parser.add_argument('-l', '--list-of-images', type=str,
                        default=f"{def_vtlm_path}/images/index.txt",
                        help='list of image locations for which features will be extracted.')

    parser.add_argument('-o', '--output-folder', type=str,
                        default=f"{def_vtlm_path}/features",
                        help='Output folder where features will be saved.')

    parser.add_argument('-f', '--format', default='pickle',
                        choices=['picklebz2', 'picklegz', 'pickle', 'npz'],
                        help='Output file format.')
    parser.add_argument('-p', '--parallel', action='store_true',
                        help='Parallel dumper process for output files.')
    parser.add_argument('-n', '--num-proposals', type=int, default=36,
                        help='Number of top proposals to accept.')
    parser.add_argument('-P', '--pool', action='store_true',
                        help='If given, average pool the conv features for each region.')

    args = parser.parse_args()

    # Setup compressor
    assert args.format.startswith(("pickle", "npz")), "Output file format unknown."

    if args.format == 'pickle':
        dump_detections, dump_suffix = fn_pickle, '.pkl'
    if args.format == 'picklegz':
        dump_detections, dump_suffix = fn_picklegz, '.pgz'
    elif args.format == 'picklebz2':
        dump_detections, dump_suffix = fn_picklebz2, '.pbz2'
    elif args.format == 'npz':
        dump_detections, dump_suffix = fn_npz, '.npz'

    if 'oid_v4' in args.model_folder:
        num_classes = 601
    else:
        print('Only oid_v4 models are supported so far')
        sys.exit(1)

    if args.parallel:
        pool = Pool(processes=3)
    else:
        pool = None

    out_folder = Path(args.output_folder) / Path(args.model_folder).name
    out_folder.mkdir(exist_ok=True, parents=True)

    pre_extraction = time.time()
    # Load image list
    image_list = read_image_list(args.list_of_images)
    if '/' not in image_list[0]:
        root = Path(args.img_root)
        image_list = [str(root / img) for img in image_list]

    print(f'Will extract features for (at most) {len(image_list)} images.')
    ds = tf.data.Dataset.from_tensor_slices(image_list)
    dataset = ds.map(
        lambda x: tf.image.decode_image(
            tf.io.read_file(x), expand_animations=False, channels=3), num_parallel_calls=2).prefetch(4)

    # build the model
    model = tf.saved_model.load(
        str(Path(args.model_folder) / 'saved_model')).signatures['serving_default']

    # Introspect num_classes to compute the class_offset for the bg-class
    class_offset = model.structured_outputs['raw_detection_scores'].shape[-1] - num_classes
    print('Class offset is: ', class_offset)

    print('Warming up model')
    model(tf.convert_to_tensor(np.ones((1, 300, 300, 3), dtype=np.uint8)))
    pre_extraction = time.time() - pre_extraction
    print(f'Setup took {pre_extraction:.3f} seconds')

    ##########
    # mainloop
    ##########
    problems = {}
    n_total = len(image_list)
    n_extracted = 0

    # Convert it to iterator
    itd = iter(dataset)
    idxs = list(range(n_total))
    for idx in tqdm.tqdm(idxs, ncols=50):
        orig_img_name = image_list[idx].split('/')[-1]
        dump_fname = str(out_folder / orig_img_name) + dump_suffix

        try:
            img = next(itd)
        except StopIteration:
            print('Iterator exhausted. Done.')
        except Exception:
            # Read failure
            problems[orig_img_name] = 'image reading error'
        else:
            if img.shape.num_elements() < 100*100*3:
                # image too small
                problems[orig_img_name] = 'too small'
            elif not os.path.exists(dump_fname):
                try:
                    dets = model(img[None, ...])
                except Exception:
                    problems[orig_img_name] = 'inference exception'
                else:
                    dets = prepare_dict(
                        dets, class_offset=class_offset, num_proposals=args.num_proposals,
                        pool=args.pool)
                    n_extracted += 1
                    if pool:
                        pool.apply_async(dump_detections, (dets, dump_fname))
                    else:
                        dump_detections(dets, dump_fname)

    if len(problems) > 0:
        randstr = str(uuid.uuid4()).split('-')[0]
        fname = str(out_folder).rstrip('/') + f'{randstr}.txt'
        with open(fname, 'w') as f:
            for img_name, prob in problems.items():
                f.write(f'{img_name}\t{prob}\n')

    print()
    print(f'# of total images requested: {n_total}')
    print(f'# of rejects/problems: {len(problems)}')
    print(f'# of newly extracted features: {n_extracted}')
