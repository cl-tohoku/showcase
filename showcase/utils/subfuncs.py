# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

import numpy as np
import torch
from logzero import logger

from showcase import constants

_root_dir = os.path.expanduser('~/.config/showcase')


def get_dataset_root(create_directory=True):
    if create_directory:
        try:
            os.makedirs(_root_dir, exist_ok=True)
        except OSError:
            if not os.path.isdir(_root_dir):
                raise
    return _root_dir


def get_config_path():
    path = Path(constants.CONFIG_PATH)
    if path.exists():
        return str(path)
    else:
        logger.warn('config.json does not exist in the showcase root dir: [{}]'.format(get_dataset_root()))
        logger.warn('Please follow instructions in README')
        raise FileNotFoundError


def get_config():
    config = json.load(open(get_config_path()))
    return config


def get_word_idx_path():
    config = get_config()
    path = Path(config['word_index_path'])
    if path.exists():
        return str(path)
    else:
        logger.warn('word index file does not exist in {}!'.format(path))
        raise FileNotFoundError


def get_pos_idx_path():
    config = get_config()
    path = Path(config['pos_index_path'])
    if path.exists():
        return str(path)
    else:
        logger.warn('pos index file does not exist in {}!'.format(path))
        raise FileNotFoundError


def get_model_path(model_type, ensemble=False):
    config = get_config()
    assert model_type in ('PREDICATE', 'ARGUMENT')

    if model_type == 'PREDICATE':
        path_dict = config['pred_model_path']
    else:
        path_dict = config['arg_model_path']

    model_paths = sorted(path_dict.items())
    if not ensemble:
        model_paths = [model_paths[0]]
    return model_paths


def load_word_idx(path_to_idx):
    word2idx = {}
    with open(path_to_idx, 'r') as fi:
        for line in fi:
            chunks = line.rstrip('\n').split('\t')
            assert len(chunks) == 2
            word = chunks[0]
            idx = int(chunks[1])
            word2idx[word] = idx
    return word2idx


def load_pretrained_word_vec():
    config = get_config()
    path = str(Path(config['word2vec_word_path']))
    embed_matrix = np.load(path)['arr_0']
    return torch.from_numpy(embed_matrix).float()


def load_pretrained_pos_vec():
    config = get_config()
    path = str(Path(config['word2vec_pos_path']))
    embed_matrix = np.load(path)['arr_0']
    return torch.from_numpy(embed_matrix).float()


def read_stdin():
    # RaSCのrequirementsに対応するためにsys.stdinでなくinput()を使う
    while True:
        try:
            yield input()
        except EOFError:
            logger.debug('EOF Found. Exit...')
            break


def predicate_info_to_pas(predicate_info):
    predicate_indices = predicate_info.nonzero()[0].tolist()
    pas = []
    for idx in predicate_indices:
        pas.append({
            'p_id': idx,
            'p_type': int(predicate_info[idx]),
            'args': [3] * len(predicate_info)
        })
    return pas
