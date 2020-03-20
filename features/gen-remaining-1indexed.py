#!/usr/bin/env python
import os

IMG_PATH = "/data/ozan/datasets/conceptual_captions/raw_files/images"

def read_extracted_files():
    extracted = []

    # 0-based filenames
    files = [f for f in os.listdir('.') if f.endswith('.extracted')]
    for fname in files:
        with open(fname) as f:
            for line in f:
                # convert to 1-based
                extracted.append(int(line.strip()) + 1)
    return set(extracted)


if __name__ == '__main__':
    # (1-based)
    extracted = read_extracted_files()

    # get all image paths (1-based)
    all_imgs = [int(f) for f in os.listdir(IMG_PATH)]

    remaining = sorted(list(set(all_imgs).difference(extracted)))

    with open('remaining', 'w') as f:
        for line in remaining:
            f.write(f'{line}\n')
