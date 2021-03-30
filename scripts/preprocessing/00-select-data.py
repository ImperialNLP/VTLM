#!/usr/bin/env python

if __name__ == '__main__':
    img_ids = []
    with open('available_features.txt') as f:
        for line in f:
            img_ids.append(int(line.strip().replace('.pkl', '')))

    pairs = []
    with open('concap_en_de_all.tsv') as f:
        for line in f:
            pairs.append(line.strip())

    selected_data = [pairs[i] for i in img_ids if '<unk>' not in pairs[i]]

    with open('concap_en_de_with_images.tsv', 'w') as f:
        for line in selected_data:
            f.write(line + '\n')

