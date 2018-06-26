# -*- coding: utf-8 -*-
import json
import subprocess
from itertools import groupby
from subprocess import PIPE

import torch
from torch import autograd
from tqdm import tqdm

from showcase.constants import UNK_WORD
from showcase.utils.subfuncs import get_config


class InputProcessor(object):
    # いい感じの基底クラス
    def __init__(self, gpu):
        self.gpu = gpu

    def create_batch(self, data):

        def batch_generator(instance, gpu):
            pas_seq = instance["pas"]
            tokens = instance["tokens"]
            pos = instance["pos"]
            predicates = list(map(lambda pas: int(pas["p_id"]), pas_seq))
            p_types = list(map(lambda pas: int(pas["p_type"]), pas_seq))
            for pas in pas_seq:
                p_id = int(pas["p_id"])
                p_type = int(pas["p_type"])
                ys = autograd.Variable(torch.LongTensor([int(a) for a in pas["args"]]))
                ws = autograd.Variable(torch.LongTensor([int(w) for w in tokens]))
                ss = autograd.Variable(torch.LongTensor([int(p) for p in pos]))
                # ps = autograd.Variable(torch.LongTensor(
                #     [p_types[predicates.index(i)] if i in predicates else 0 for i in range(len(tokens))]))
                # ts = autograd.Variable(torch.LongTensor([p_type if i == p_id else 0 for i in range(len(tokens))]))

                ps = torch.zeros(len(tokens), 2)
                ts = torch.zeros(len(tokens), 2)
                for i in range(len(tokens)):
                    if i in predicates:
                        ps[i][p_types[predicates.index(i)] - 1] = 1
                    if i == p_id:
                        ts[i][p_type - 1] = 1
                ps = autograd.Variable(ps)
                ts = autograd.Variable(ts)

                if gpu >= 0:
                    ys = ys.cuda(gpu)
                    ws = ws.cuda(gpu)
                    ss = ss.cuda(gpu)
                    ps = ps.cuda(gpu)
                    ts = ts.cuda(gpu)
                yield [[ws, ss, ps, ts], ys]

        for sentence in data:
            if sentence["pas"]:
                instances = list(batch_generator(sentence, self.gpu))
                xss, yss = zip(*instances)
                yield [list(map(lambda e: torch.stack(e), zip(*xss))), torch.stack(yss)]

    def batch_for_pred_detection(self, instances, batch_size):
        instances = sorted(instances, key=lambda instance: len(instance["tokens"]))

        batches = []
        for (k, g) in groupby(instances, key=lambda instance: len(instance["tokens"])):
            batches += self.e2e_batch_grouped_by_sentence_length_for_pred(list(g), batch_size, self.gpu)

        return batches

    # this is batchgenerator for predicate classification
    @staticmethod
    def e2e_batch_grouped_by_sentence_length_for_pred(instances, batch_size, gpu):
        remaining = 0 if len(instances) % batch_size == 0 else 1

        num_batch = len(instances) // batch_size + remaining

        for i in range(num_batch):
            start = i * batch_size
            b = instances[start:start + batch_size]
            xs = []
            ys = []
            for instance in b:
                pas_seq = instance["pas"]
                tokens = autograd.Variable(torch.LongTensor([int(t) for t in instance["tokens"]]))
                pos = autograd.Variable(torch.LongTensor([int(p) for p in instance["pos"]]))
                ps = autograd.Variable(torch.LongTensor([0 for i in range(len(tokens))]))

                for pas in pas_seq:
                    p_id = int(pas["p_id"])
                    p_type = int(pas["p_type"])
                    ps[p_id] = p_type

                if gpu >= 0:
                    tokens = tokens.cuda(gpu)
                    pos = pos.cuda(gpu)
                    ps = ps.cuda(gpu)

                xs.append([tokens, pos])
                ys.append(ps)
            yield [list(map(lambda e: torch.stack(e), zip(*xs))), torch.stack(ys)]

    def convert(self, paragraph, word2idx):
        raise NotImplementedError


class RawTextProcessor(InputProcessor):
    def __init__(self, gpu):
        super(RawTextProcessor, self).__init__(gpu)

    def convert(self, paragraph, word2idx, pos2idx):
        cabocha_results, idx2cabocha_result_list = self._apply_cabocha(paragraph)
        model_inputs = [self._cabocha_to_json(x, word2idx, pos2idx) for x in cabocha_results]
        return cabocha_results, idx2cabocha_result_list, model_inputs

    def __words_to_indices(self, words, word2idx, pos2idx):
        # words: cabochaの解析結果からメタ情報を除いたもの
        word_indices = []
        pos_indices = []
        for i, word in enumerate(words):
            surface, details = word.strip().split('\t')

            # base formの獲得
            # 「サ変名詞 + する」の場合は，モデルへの入力を「サ変名詞 + サ変名詞する」に置換する
            if i != 0 and details.startswith('動詞') and words[i - 1].split('\t')[-1].split(',')[1] == 'サ変名詞':
                current_base = details.split(',')[-3]
                prev_base = words[i - 1].split('\t')[0]
                base = prev_base + current_base
            else:
                base = details.split(',')[-3]

            # 品詞細分類の獲得
            pos_detail = '-'.join(details.split(',')[:3])

            if base in word2idx:
                word_indices.append(word2idx[base])
            elif pos_detail in word2idx:
                word_indices.append(word2idx[pos_detail])
            else:
                word_indices.append(word2idx[UNK_WORD])

            # posの場合はUNKへのback-off処理を書けない...(wikipediaが全ての品詞細分類をカバーしているという仮定が存在)
            # でもとりあえず書いちゃう!!!
            if pos_detail in pos2idx:
                pos_indices.append(pos2idx[pos_detail])
            else:
                pos_indices.append(0)  # 諦める
        return word_indices, pos_indices

    def _cabocha_to_json(self, cabocha_result, word2idx, pos2idx):
        # remove meta info from cabocha output
        filtered_cabocha = [x for x in cabocha_result if not x.startswith('* ')]
        word_indices, pos_indices = self.__words_to_indices(filtered_cabocha, word2idx, pos2idx)
        dummy_pas = [
            {
                'p_id': 0,
                'p_type': 1
            }
        ]
        out = {
            'tokens': word_indices,
            'pos': pos_indices,
            'pas': dummy_pas
        }
        return out

    def _apply_cabocha(self, paragraph):
        # 生テキストにCabochaを適用する
        config = get_config()
        cabocha = config['cabocha']['cabocha_path']
        cabocha_dict = config['cabocha']['juman_dic_path']
        cabocha_dep_model = config['cabocha']['juman_dep_model_path']
        cabocha_chunk_model = config['cabocha']['juman_chunk_model_path']
        with subprocess.Popen(
                '{} -f1 -P JUMAN -d {} -m {} -M {}'.format(
                    cabocha, cabocha_dict,
                    cabocha_dep_model, cabocha_chunk_model
                ),
                shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                universal_newlines=True) as pipe:
            sent = '\n'.join(x.strip() for x in paragraph)
            out, err = pipe.communicate(sent.strip())
            cabocha_results = []
            idx2cabocha_result_list = []
            for is_eos, section in groupby(out.strip().split('\n'), key=lambda x: x.strip() == 'EOS'):
                if not is_eos:
                    sentence_result = list(section)

                    token_idx_to_cabocha_idx = {}
                    i = 0
                    for n, line in enumerate(sentence_result):
                        if not line.startswith('*'):
                            token_idx_to_cabocha_idx[i] = n
                            i += 1
                    idx2cabocha_result_list.append(token_idx_to_cabocha_idx)
                    cabocha_results.append(sentence_result)
        return cabocha_results, idx2cabocha_result_list


def end2end_dataset(data_path, data_rate):
    data = open(data_path).readlines()
    len_data = len(data)
    data = data[:int(len_data * data_rate / 100)]
    return [json.loads(line.strip()) for line in tqdm(data, mininterval=5) if line.strip()]
