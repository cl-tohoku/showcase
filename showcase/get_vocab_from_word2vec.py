# -*- coding: utf-8 -*-
"""

"""
import sys


def main(fi):
    vocab = set()
    for n, line in enumerate(fi):
        if n == 0:
            continue
        vocab.add(line.split()[0])

    for n, token in enumerate(vocab):
        print('{}\t{}'.format(token, n))


if __name__ == "__main__":
    main(sys.stdin)
