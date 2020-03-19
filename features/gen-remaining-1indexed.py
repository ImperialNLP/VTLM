#!/usr/bin/env python
import os

IMG_PATH = "/data/ozan/datasets/conceptual_captions/raw_files/images"

if __name__ == '__main__':
    extracted = []
    # Hacettepe finished these
    with open('extracted_images') as f:
        for line in f:
            # convert 0-indices to 1-indices
            extracted.append(int(line.strip()) + 1)

    # get all image paths (starts at 1 not 0)
    all_imgs = [int(f) for f in os.listdir(IMG_PATH)]
    remaining = sorted(list(set(all_imgs).difference(set(extracted))))

    with open('remaining', 'w') as f:
        for line in remaining:
            f.write(f'{IMG_PATH}/{line}\n')
