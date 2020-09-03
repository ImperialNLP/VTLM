#!/usr/bin/env python
import sys


if __name__ == '__main__':
    try:
        words = sys.argv[1]
    except IndexError as ie:
        print(f'Usage: {sys.argv[0]} <word list>')
        sys.exit(1)

    word_list = {}
    with open(words) as f:
        for line in f:
            word_list[line.strip()] = '<special1>'


    for line in sys.stdin:
        words = line.strip().split()
        sent = ' '.join(map(lambda x: word_list.get(x, x), words))
        print(sent)

