# -*- coding: utf-8 -*-
import os

GA_ID = 0  # ガ格
WO_ID = 1  # ヲ格
NI_ID = 2  # ニ格

id2arg_type = {
    GA_ID: 'ga',
    WO_ID: 'wo',
    NI_ID: 'ni'
}

PRED_ID = 1  # 普通の述語
NOUN_ID = 2  # 名詞述語

id2pred_type = {
    PRED_ID: 'pred',
    NOUN_ID: 'noun'
}

UNK_WORD = '__OOV__'

CONFIG_PATH = os.path.join(os.environ.get('HOME'), '.config', 'showcase', 'config.json')
