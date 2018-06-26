# -*- coding: utf-8 -*-
"""
入力: word2vecから獲得した単語ベクトルファイル(.txt) & 品詞細分類を含めた全語彙ファイル(1行1vocab)
出力: 単語ベクトルファイルをnpyファイルにしたもの．全語彙ファイルに登場しないものについては適当に初期化．
"""
import argparse
import os
from itertools import islice
from pathlib import Path

import numpy as np


def main(args):
    # load vocabulary file
    word_to_index = {l.strip().split('\t')[0]: int(l.strip().split('\t')[1]) for l in open(args.vocab, 'r')}

    # set word2vec vector to the matrix
    array = np.random.uniform(low=-0.05, high=0.05, size=(len(word_to_index), args.dim))
    with open(args.word2vec, 'r') as fi:
        for line in islice(fi, 1, None):
            fields = line.rstrip().split(' ')
            assert len(fields) == (args.dim + 1)

            token = fields[0]
            vec = np.asarray(fields[1::], dtype=np.float32)
            if token in word_to_index:
                array[word_to_index[token]] = vec

    # save matrix
    path = Path(args.word2vec)
    dest = str(Path(path.parent, path.stem))
    np.savez_compressed(dest, array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Word2Vec')
    parser.add_argument('--word2vec', type=os.path.abspath, help='word2vec embedding file (*.txt format)')
    parser.add_argument('--vocab', type=os.path.abspath, help='vocabulary file')
    parser.add_argument('--dim', type=int, help='dimension')
    args = parser.parse_args()
    main(args)
