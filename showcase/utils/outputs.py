# -*- coding: utf-8 -*-
import json
from collections import defaultdict

from showcase.constants import id2arg_type, NI_ID, WO_ID, GA_ID, id2pred_type, PRED_ID, NOUN_ID


def print_pretty(result, arguments, jsonl):
    # 構造化されていない，human-readableな出力
    lines = [x for x in result if not x.startswith('* ')]
    tokens = [x.split('\t')[0] for x in lines]
    out_text = []
    out_text.append(' '.join(tokens))  # wakati text
    for pas, arg in zip(jsonl['pas'], arguments):
        out_text.append('Predicate: {}'.format(tokens[pas['p_id']]))
        out_text.append('Predicate Type: {}'.format(id2pred_type[pas['p_type']]))
        pas_text = []
        for n, a in enumerate(arg):
            if a == -1:
                token = 'n/a'
            else:
                token = tokens[a]
            pas_text.append('{}: {}'.format(id2arg_type[n], token))
        out_text.append('\t'.join(pas_text))
    return '\n'.join(out_text) + '\n', 0


def print_json(result, arguments, jsonl, rasc=False):
    # json formatでの出力
    # TODO: 同じsurfaceの単語に対応（曖昧性の解消が必要）
    lines = [x for x in result if not x.startswith('* ')]
    tokens = [x.split('\t')[0] for x in lines]
    out = {
        'wakati': tokens,
        'pas': []
    }
    for pas, arg in zip(jsonl['pas'], arguments):
        out['pas'].append(
            {
                'predicate_idx': pas['p_id'],
                'predicate_type': id2pred_type[pas['p_type']],
                'ga_idx': int(arg[GA_ID]),
                'o_idx': int(arg[WO_ID]),
                'ni_idx': int(arg[NI_ID])
            }
        )
    if not rasc:
        return json.dumps(out, ensure_ascii=False), 0
    else:
        return out, 0


def print_cabocha(result, arguments, jsonl, idx_hash, init_id, rasc=False):
    # NAIST Text Corpusに従うformat
    pas_info = defaultdict(list)
    for pas, arg in zip(jsonl['pas'], arguments):
        pred = pas['p_id']
        pas_info[idx_hash[pred]].append('type="{}"'.format(id2pred_type[pas['p_type']]))
        for n, a in enumerate(arg):
            if a > 0:
                pas_info[idx_hash[a]].append('id="{}"'.format(init_id))
                pas_info[idx_hash[pred]].append('{}="{}"'.format(id2arg_type[n], init_id))
                init_id += 1
    out_text = []
    for n, line in enumerate(result):
        if n in pas_info:
            pas = ' '.join(pas_info[n])
        else:
            pas = ''
        out_text.append('{}\t{}'.format(line, pas))
    else:
        if not rasc:
            out_text.append('EOS')
    return '\n'.join(out_text), init_id


def print_conll(result, arguments, jsonl, idx_hash, init_id, rasc=False):
    # conll format

    def cabocha_line_to_conll(line):
        surface, details = line.split('\t')
        details = details.split(',')
        base = details[-3]
        pos = details[0]
        detail_pos = '{}-{}'.format(details[0], details[1]) if details[1] != '*' else details[0]
        return '{}\t{}\t{}\t{}'.format(surface, base, pos, detail_pos)

    pas_list = []
    for n, (pas, arg) in enumerate(zip(jsonl['pas'], arguments)):
        pred = pas['p_id']
        lis = ['*'] * len(idx_hash)
        lis[pred] = id2pred_type[pas['p_type']]
        if arg[GA_ID] != -1:
            lis[arg[GA_ID]] = 'ARG_GA'
        if arg[WO_ID] != -1:
            lis[arg[WO_ID]] = 'ARG_WO'
        if arg[NI_ID] != -1:
            lis[arg[NI_ID]] = 'ARG_NI'
        pas_list.append(lis)
        init_id += 1

    out_text = []
    for (n, idx), pas_elem in zip(sorted(idx_hash.items()), list(zip(*pas_list))):
        cabocha_conll = cabocha_line_to_conll(result[idx])
        pas_conll = '\t'.join(pas_elem)
        out_text.append('{}\t{}\t{}'.format(n+1, cabocha_conll, pas_conll))
    else:
        if not rasc:
            out_text.append('EOS')
    return '\n'.join(out_text), init_id
