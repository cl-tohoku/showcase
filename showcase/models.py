import json
import logging
import math
import os
from itertools import groupby
from pathlib import Path

import logzero
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger
from torch import autograd

from showcase.layer.attention import BatchAttentionLayer
from showcase.layer.bi_gru import BiGRU
from showcase.utils import outputs
from showcase.utils.inputs import RawTextProcessor
from showcase.utils.subfuncs import get_word_idx_path, get_model_path, get_dataset_root, get_config, load_word_idx, \
    load_pretrained_word_vec, predicate_info_to_pas, get_pos_idx_path, load_pretrained_pos_vec


def prob_to_argument(probs):
    probs = list(map(list, zip(*probs)))
    lis = []
    for prob in probs:
        avg_prob = sum(prob) / len(prob)
        query = avg_prob.argsort(axis=0)[::-1][0]
        probs = avg_prob[query, np.arange(query.shape[0])]

        lis.append([y if x > 0 else -1 for x, y in zip(probs, query)])
    return lis


def prob_to_predicate(probs):
    return np.average(probs, axis=0).argmax(axis=1)


class E2EStackedBiRNNMpPoolAttention(nn.Module):
    def __init__(self, dim_u: int, dim_pool: int,
                 depth: int,
                 dim_out: int,
                 vocab_size: int,
                 dim_embed: int,
                 drop_u: int, ensemble_mode: bool = False):
        super(E2EStackedBiRNNMpPoolAttention, self).__init__()

        self.ensemble_mode = ensemble_mode
        self.vocab_size = vocab_size
        self.embedding_dim = dim_embed
        self.dim_u = dim_u
        self.dim_pool = dim_pool
        self.depth = depth

        self.word_emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

        # input is a set of embedding and predicate marker
        self.gru = BiGRU(self.embedding_dim + 2, dim_u, num_layers=depth, dropout=drop_u)

        self.linear = nn.Linear(dim_u * 2, dim_pool)
        self.linear2 = nn.Linear(dim_pool * 2, dim_pool)

        self.attention = BatchAttentionLayer(dim_pool, dim_pool)

        self.output_layer = nn.Linear(dim_pool, dim_out)

    def _is_model_on_gpu(self):
        return next(self.parameters()).is_cuda

    def forward(self, x):
        words, is_pred, is_target = x
        if self.ensemble_mode and self._is_model_on_gpu():
            words = autograd.Variable(words, volatile=True).cuda()
            is_pred = autograd.Variable(is_pred, volatile=True).cuda()
            is_target = autograd.Variable(is_target, volatile=True).cuda()
        elif not self.ensemble_mode and self._is_model_on_gpu():
            words = autograd.Variable(words).cuda()
            is_pred = autograd.Variable(is_pred).cuda()
            is_target = autograd.Variable(is_target).cuda()
        else:
            words = autograd.Variable(words)
            is_pred = autograd.Variable(is_pred)
            is_target = autograd.Variable(is_target)

        embeds = self.word_emb(words)
        inputs = torch.cat([embeds, is_pred, is_target], dim=2)
        gru_outputs = self.gru(inputs)  # output: (pred_batch, word, emb)

        batch_size = gru_outputs.size(0)
        token_size = gru_outputs.size(1)
        pooling = nn.MaxPool1d(batch_size)

        outputs = []
        for batch_elm_i in range(batch_size):
            # combine the prediction for a word and a target predicate with that for another predicate
            target_expr = gru_outputs[batch_elm_i]
            combined = [torch.cat([target_expr, gru_outputs[batch_elm_j]], dim=1) for batch_elm_j in range(batch_size)]
            hidden = F.relu(self.linear(torch.stack(combined)))  # (batch, words, dim_u*4)
            hidden = torch.transpose(hidden, 0, 2)
            hidden = torch.transpose(pooling(hidden), 0, 2)[0]
            outputs += [hidden]

        outputs = torch.stack(outputs)

        # get a target word
        # for each predicate pi, for each word wj, get an attention over words: (pred, word, emb)
        attend = torch.stack([self.attention.forward(outputs[pred_i], outputs[pred_i]) for pred_i in range(batch_size)])
        combined = torch.cat([outputs, attend], dim=2)

        hidden = self.linear2(combined)  # (words, preds, emb)
        outputs = F.relu(hidden)  # (preds, words, emb)

        return [F.log_softmax(self.output_layer(out), dim=1) for out in outputs]

    def predict(self, xss, thres):
        self.eval()
        scores = self(xss)
        lis = []
        for score, in zip(scores):
            predicted = torch.pow(torch.zeros(score.size()) + math.e, score.cpu().data).numpy()[:, :3]
            pred_bias = predicted - np.array(thres)
            lis.append(pred_bias)
        return lis


class WordEmb(nn.Module):
    def __init__(self, embedding_matrix):
        super(WordEmb, self).__init__()
        self.vocab_size = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)

        self.word_emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_emb.weight = nn.Parameter(embedding_matrix)
        self.word_emb.weight.requires_grad = False

    def forward(self, x):
        # words, pos = x

        # word_embeds = self.word_emb(words)
        # pos_embeds = self.word_emb(pos)
        # return torch.cat([word_embeds, pos_embeds], dim=2)
        return self.word_emb(x)


class E2EStackedBiRNNMpPoolAttentionWithoutWordEmb(nn.Module):
    """  using same weights for pooling layers.
    """

    def __init__(self, dim_u: int, dim_pool: int,
                 depth: int,
                 dim_in: int,
                 dim_out: int,
                 ensemble_mode: bool = False):
        super(E2EStackedBiRNNMpPoolAttentionWithoutWordEmb, self).__init__()

        self.ensemble_mode = ensemble_mode
        self.dim_u = dim_u
        self.dim_input = dim_in
        self.dim_pool = dim_pool
        self.depth = depth

        # input is a set of embedding and predicate marker
        self.gru = BiGRUForSRL(dim_in, dim_u, num_layers=depth)

        self.linear = nn.Linear(dim_u * 2, dim_pool)
        self.linear2 = nn.Linear(dim_pool * 2, dim_pool)

        self.attention = BatchAttentionLayer(dim_pool, dim_pool)

        self.output_layer = nn.Linear(dim_pool, dim_out)

    def forward(self, x):
        """
        PAS for each predicate is combined in a single batch.
        """
        embeds, is_pred, is_target = x

        inputs = torch.cat([embeds, is_pred, is_target], dim=2)
        gru_outputs = self.gru(inputs)  # output: (pred_batch, word, emb)

        batch_size = gru_outputs.size(0)
        pooling = nn.MaxPool1d(batch_size)

        outputs = []
        for batch_elm_i in range(batch_size):
            # combine the prediction for a word and a target predicate with that for another predicate
            target_expr = gru_outputs[batch_elm_i]
            combined = [torch.cat([target_expr, gru_outputs[batch_elm_j]], dim=1) for batch_elm_j in range(batch_size)]
            hidden = F.relu(self.linear(torch.stack(combined)))  # (batch, words, dim_u*4)
            hidden = torch.transpose(hidden, 0, 2)
            hidden = torch.transpose(pooling(hidden), 0, 2)[0]
            outputs += [hidden]

        outputs = torch.stack(outputs)

        # get a target word
        # for each predicate pi, for each word wj, get an attention over words: (pred, word, emb)
        attend = torch.stack([self.attention.forward(outputs[pred_i], outputs[pred_i]) for pred_i in range(batch_size)])
        combined = torch.cat([outputs, attend], dim=2)

        hidden = self.linear2(combined)  # (words, preds, emb)
        outputs = F.relu(hidden)  # (preds, words, emb)

        return [F.log_softmax(self.output_layer(out), dim=1) for out in outputs]

    def get_attention(self, x):
        """
        PAS for each predicate is combined in a single batch.
        """
        words, is_pred, is_target = x
        # if self.ensemble_mode and torch.cuda.is_available():
        #     words = autograd.Variable(words, volatile=True).cuda()
        #     is_pred = autograd.Variable(is_pred, volatile=True).cuda()
        #     is_target = autograd.Variable(is_target, volatile=True).cuda()
        # elif (not self.ensemble_mode) and torch.cuda.is_available():
        #     words = autograd.Variable(words).cuda()
        #     is_pred = autograd.Variable(is_pred).cuda()
        #     is_target = autograd.Variable(is_target).cuda()
        # else:
        #     words = autograd.Variable(words)
        #     is_pred = autograd.Variable(is_pred)
        #     is_target = autograd.Variable(is_target)

        embeds = self.word_emb(words)
        inputs = torch.cat([embeds, is_pred, is_target], dim=2)
        gru_outputs = self.gru(inputs)  # output: (pred_batch, word, emb)

        batch_size = gru_outputs.size(0)
        token_size = gru_outputs.size(1)
        pooling = nn.MaxPool1d(batch_size)

        outputs = []
        for batch_elm_i in range(batch_size):
            # combine the prediction for a word and a target predicate with that for another predicate
            target_expr = gru_outputs[batch_elm_i]
            combined = [torch.cat([target_expr, gru_outputs[batch_elm_j]], dim=1) for batch_elm_j in range(batch_size)]
            hidden = F.relu(self.linear(torch.stack(combined)))  # (batch, words, dim_u*4)
            hidden = torch.transpose(hidden, 0, 2)
            hidden = torch.transpose(pooling(hidden), 0, 2)[0]
            outputs += [hidden]

        outputs = torch.stack(outputs)

        attend = torch.stack(
            [self.attention.get_attention(outputs[pred_i], outputs[pred_i]) for pred_i in range(batch_size)])
        return attend


# これを使う
class E2EStackedBiRNNMpPoolAttentionFixedWordEmb(nn.Module):
    def __init__(self, dim_u: int, dim_pool: int, depth: int,
                 dim_out: int,
                 word_embedding_matrix, pos_embedding_matrix,
                 drop_u: int):
        super(E2EStackedBiRNNMpPoolAttentionFixedWordEmb, self).__init__()
        self.word_emb = WordEmb(word_embedding_matrix)
        self.pos_emb = WordEmb(pos_embedding_matrix)
        # self.pred_emb = nn.Embedding(3, 8)

        self.bi_gru = E2EStackedBiRNNMpPoolAttentionWithoutWordEmb(dim_u, dim_pool, depth,
                                                                   self.word_emb.embedding_dim + self.pos_emb.embedding_dim + 4,
                                                                   dim_out)

    def forward(self, x):
        words, pos, is_pred, is_target = x
        # if torch.cuda.is_available():
        #     words = autograd.Variable(words).cuda()
        #     pos = autograd.Variable(pos).cuda()
        #     is_pred = autograd.Variable(is_pred).cuda()
        #     is_target = autograd.Variable(is_target).cuda()
        # else:
        #     words = autograd.Variable(words)
        #     pos = autograd.Variable(pos)
        #     is_pred = autograd.Variable(is_pred)
        #     is_target = autograd.Variable(is_target)

        w_emb = self.word_emb(words)
        p_emb = self.pos_emb(pos)
        emb = torch.cat([w_emb, p_emb], dim=2)
        # ps_emb = self.pred_emb(is_pred)
        # t_emb = self.pred_emb(is_target)
        output = self.bi_gru([emb, is_pred, is_target])
        return output

    def save(self, file):
        torch.save(self.bi_gru.state_dict(), file)

    def load(self, file):
        self.bi_gru.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))

    def predict(self, xss, thres):
        self.eval()
        scores = self(xss)
        lis = []
        for score, in zip(scores):
            predicted = torch.pow(torch.zeros(score.size()) + math.e, score.cpu().data).numpy()[:, :3]
            pred_bias = predicted - np.array(thres)
            lis.append(pred_bias)
        return lis


class E2EStackedBiRNNwithoutWordEmb(nn.Module):
    def __init__(self, dim_u: int, depth: int,
                 dim_out: int,
                 input_dim: int,
                 drop_u: int):
        super(E2EStackedBiRNNwithoutWordEmb, self).__init__()

        self.imput_dim = input_dim
        self.hidden_dim = dim_u
        self.depth = depth

        # input is a set of embedding and predicate marker
        self.gru = BiGRUForSRL(input_dim, dim_u, num_layers=depth, dropout=drop_u)
        self.output_layer = nn.Linear(dim_u, dim_out)

    def forward(self, x):
        words, is_target = x

        inputs = torch.cat([words, is_target], dim=2)
        outputs = self.gru(inputs)

        return [F.log_softmax(self.output_layer(out), dim=1) for out in outputs]


class E2EStackedBiRNNFixedWordEmb(nn.Module):
    def __init__(self, dim_u: int, depth: int,
                 dim_out: int,
                 embedding_matrix,
                 drop_u: int):
        super(E2EStackedBiRNNFixedWordEmb, self).__init__()
        self.emb = WordEmb(embedding_matrix)
        self.bi_gru = E2EStackedBiRNNwithoutWordEmb(dim_u, depth, dim_out,
                                                    self.emb.embedding_dim * 2 + 1, drop_u)

    def forward(self, x):
        words, pos, is_target = x
        # if torch.cuda.is_available():
        #     words = autograd.Variable(words).cuda()
        #     pos = autograd.Variable(pos).cuda()
        #     is_target = autograd.Variable(is_target).cuda()
        # else:
        #     words = autograd.Variable(words)
        #     pos = autograd.Variable(pos)
        #     is_target = autograd.Variable(is_target)

        emb = self.emb([words, pos])
        output = self.bi_gru([emb, is_target])

        return output

    def save(self, file):
        torch.save(self.bi_gru.state_dict(), file)

    def load(self, file):
        self.bi_gru.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))


# これは使わない
class E2EStackedBiRNNForPredicateDetection(nn.Module):
    def __init__(self, dim_u: int, depth: int,
                 dim_out: int,
                 embedding_matrix,
                 drop_u: int):
        super(E2EStackedBiRNNForPredicateDetection, self).__init__()
        self.emb = WordEmb(embedding_matrix)
        self.bi_gru = BiGRU4PredicateDetection(dim_u, depth, dim_out,
                                               self.emb.embedding_dim * 2, drop_u)

    def forward(self, x):
        words, pos = x
        # if torch.cuda.is_available():
        #     words = autograd.Variable(words).cuda()
        #     pos = autograd.Variable(pos).cuda()
        # else:
        #     words = autograd.Variable(words)
        #     pos = autograd.Variable(pos)

        emb = self.emb([words, pos])
        output = self.bi_gru(emb)

        return output

    def save(self, file):
        torch.save(self.bi_gru.state_dict(), file)

    def load(self, file):
        self.bi_gru.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))


class BiGRU4PredicateDetection(nn.Module):
    def __init__(self, dim_u: int, depth: int,
                 dim_out: int,
                 input_dim: int,
                 drop_u: int):
        super(BiGRU4PredicateDetection, self).__init__()

        self.imput_dim = input_dim
        self.hidden_dim = dim_u
        self.depth = depth

        self.gru = BiGRUForSRL(input_dim, dim_u, num_layers=depth, dropout=drop_u)
        self.output_layer = nn.Linear(dim_u, dim_out)

    def forward(self, x):
        words = x
        outputs = self.gru(words)

        return [F.log_softmax(self.output_layer(out), dim=1) for out in outputs]


class BiGRUSelfAtt4PredicateDetection(nn.Module):
    def __init__(self, dim_u: int, depth: int,
                 dim_pool: int,
                 dim_out: int,
                 input_dim: int):
        super(BiGRUSelfAtt4PredicateDetection, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = dim_u
        self.depth = depth

        self.gru = BiGRUForSRL(input_dim, dim_u, num_layers=depth)
        self.output_layer = nn.Linear(dim_u, dim_out)
        self.attention = BatchAttentionLayer(dim_u, dim_u)
        self.linear = nn.Linear(dim_u * 2, dim_pool)
        self.output_layer = nn.Linear(dim_pool, dim_out)

    def forward(self, x):
        words = x
        gru_outputs = self.gru(words)
        batch_size = gru_outputs.size(0)

        combined = []
        for batch_elm_i in range(batch_size):
            target_expr = gru_outputs[batch_elm_i]
            combined += [torch.cat([target_expr, self.attention.forward(target_expr, target_expr)], dim=1)]
        outputs = F.relu(self.linear(torch.stack(combined)))  # (batch, words, dim_u*4)

        return [F.log_softmax(self.output_layer(out), dim=1) for out in outputs]


# これを使う
class E2EStackedBiRNNSelfAttentionForPredicateDetection(nn.Module):
    def __init__(self, dim_u: int, dim_pool: int,
                 depth: int, dim_out: int,
                 word_embedding_matrix, pos_embedding_matrix):
        super(E2EStackedBiRNNSelfAttentionForPredicateDetection, self).__init__()
        self.word_emb = WordEmb(word_embedding_matrix)
        self.pos_emb = WordEmb(pos_embedding_matrix)
        self.bi_gru = BiGRUSelfAtt4PredicateDetection(dim_u, depth, dim_pool, dim_out,
                                                      self.word_emb.embedding_dim + self.pos_emb.embedding_dim)
        # input is a set of embedding and predicate marker

    def forward(self, x):
        words, pos = x
        w_emb = self.word_emb(words)
        p_emb = self.pos_emb(pos)
        emb = torch.cat([w_emb, p_emb], dim=2)
        outputs = self.bi_gru(emb)

        return outputs

    def save(self, file):
        torch.save(self.bi_gru.state_dict(), file)

    def load(self, file):
        self.bi_gru.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))

    def predict(self, xss):
        self.eval()
        logits = self(xss)[0]
        probs = np.exp(logits.data.cpu().numpy())
        return probs


class BiGRUForSRL(nn.Module):
    def __init__(self, dim_in: int, dim_u: int, num_layers: int):
        super(BiGRUForSRL, self).__init__()

        self.dim_in = dim_u
        self.dim_u = dim_u
        self.depth = num_layers

        # input is a set of embedding and predicate marker
        self.gru_in = nn.GRU(dim_in, dim_u, num_layers=1, batch_first=True)
        self.grus = nn.ModuleList([nn.GRU(dim_u, dim_u, num_layers=1, batch_first=True) for _ in
                                   range(num_layers - 1)])

    def forward(self, x):
        out, _ = self.gru_in(x)
        for gru in self.grus:
            flipped = self.reverse(out.transpose(0, 1)).transpose(0, 1)
            output, _ = gru(flipped)
            out = flipped + output

        return self.reverse(out.transpose(0, 1)).transpose(0, 1)

    def reverse(self, x):
        idx = torch.arange(x.size(0) - 1, -1, -1).long()
        idx = torch.LongTensor(idx)
        if x.data.is_cuda:
            idx = idx.cuda(x.get_device())
        return x[idx]


class ModelAPI(object):
    def __init__(self, in_format, out_format, ensemble, debug, gpu):
        if in_format == 'raw':
            self.processor = RawTextProcessor(gpu)
        else:
            raise NotImplementedError
        self.ensemble = ensemble
        self.logger = logger
        self.pred_models = []
        self.arg_models = []
        self.arg_thresholds = []
        self.word2idx = None
        self.idx2word = None
        self.init_id = 0
        self.out_format = out_format
        self.debug = debug
        self.gpu = gpu

        config = get_config()
        self.embed_dim = config['model']['embed_dim']
        self.rnn_dim = config['model']['rnn_dim']
        self.pooling_dim = config['model']['pooling_dim']
        self.rnn_depth = config['model']['rnn_depth']
        self.threshold_dict = config['model_threshold']

    def set_log_file(self):
        if self.debug:
            logzero.loglevel(logging.DEBUG)

        log_name = 'log.txt'
        log_path = str(Path(get_dataset_root(), 'log'))
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        dest = str(Path(log_path, log_name))
        logzero.logfile(dest)
        self.logger.debug('Log destination: [{}]'.format(dest))

    def prepare_model(self):
        self.logger.debug('Loading Vocabulary File')
        self.word2idx = load_word_idx(get_word_idx_path())
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.logger.debug('Vocabulary Size: {}'.format(len(self.word2idx)))

        self.logger.debug('Loading Pos Vocabulary File')
        self.pos2idx = load_word_idx(get_pos_idx_path())
        self.idx2pos = {v: k for k, v in self.pos2idx.items()}
        self.logger.debug('Pos Vocabulary Size: {}'.format(len(self.pos2idx)))

        self.logger.debug('Loading Word Vectors')
        embed_mat = load_pretrained_word_vec()
        self.logger.debug('Embedding Shape: {}'.format(embed_mat.shape))

        self.logger.debug('Loading Pos Vectors')
        pos_embed_mat = load_pretrained_pos_vec()
        self.logger.debug('Pos Embedding Shape: {}'.format(pos_embed_mat.shape))

        if self.ensemble:
            self.logger.debug('Ensemble Mode')
        else:
            self.logger.debug('Single Model Mode')

        # Prepare Predicate Detector
        for name, path in get_model_path(model_type='PREDICATE', ensemble=self.ensemble):
            model = E2EStackedBiRNNSelfAttentionForPredicateDetection(self.rnn_dim, self.pooling_dim, self.rnn_depth, 3,
                                                                      word_embedding_matrix=embed_mat,
                                                                      pos_embedding_matrix=pos_embed_mat)
            self.logger.debug('Loading params of model [{}] from [{}]'.format(name, path))
            model.load(path)
            model.eval()
            self.pred_models.append(model)
        self.logger.debug('Params has been loaded for Predicate Detector')

        # Prepare Argument Detector
        for name, path in get_model_path(model_type='ARGUMENT', ensemble=self.ensemble):
            model = E2EStackedBiRNNMpPoolAttentionFixedWordEmb(self.rnn_dim, self.pooling_dim, self.rnn_depth, 4,
                                                               word_embedding_matrix=embed_mat,
                                                               pos_embedding_matrix=pos_embed_mat, drop_u=0.0)
            self.logger.debug('Loading params of model [{}] from [{}]'.format(name, path))
            model.load(path)
            model.eval()
            self.arg_models.append(model)

            self.arg_thresholds.append(self.threshold_dict[name])
        self.logger.debug('Params has been loaded for Argument Detector')

        # Send Models to GPU
        gpu_id = self.gpu
        if torch.cuda.is_available() and gpu_id >= 0:
            for n, model in enumerate(self.pred_models):
                self.logger.debug('Sending Predicate Detector #{:d} to GPU {}'.format(n, gpu_id))
                model.cuda()

            for n, model in enumerate(self.arg_models):
                self.logger.debug('Sending Argument Detector #{:d} to GPU {}'.format(n, gpu_id))
                model.cuda()

    def infer(self, fi):
        for is_empty, paragraph in groupby(fi, key=lambda x: x.strip() == ''):
            if is_empty:
                continue

            # とりあえず生テキストを入れた場合を想定
            # 色んな抽象化の仕方が考えられるが，result, idx_hash, model_inputsは常に作るという設計にする
            # gold predを与える場合はcabocha inputを使うか...
            result, idx_hash, model_inputs = self.processor.convert(paragraph, self.word2idx, self.pos2idx)

            # Predicate Detection
            for n, jsonl in enumerate(model_inputs):
                xss, yss = self.processor.batch_for_pred_detection([jsonl], batch_size=1)[0]
                probs = [model.predict(xss) for model in self.pred_models]
                predicates = prob_to_predicate(probs)
                jsonl['pas'] = predicate_info_to_pas(predicates)

            # Argument Detection
            self.init_id = 0
            for n, jsonl in enumerate(model_inputs):
                if len(jsonl['pas']) > 0:  # if there exists a predicate...
                    xss, yss = list(self.processor.create_batch([jsonl]))[0]
                    probs = [model.predict(xss, thresh) for model, thresh in zip(self.arg_models, self.arg_thresholds)]
                    arguments = prob_to_argument(probs)
                else:
                    arguments = []
                self.print_result(n, result, arguments, jsonl, idx_hash)

    def print_result(self, n, results, arguments, jsonl, idx_hash):
        if self.out_format == 'cabocha':
            out_text, ids = outputs.print_cabocha(results[n], arguments, jsonl, idx_hash[n], self.init_id)
        elif self.out_format == 'pretty':
            out_text, ids = outputs.print_pretty(results[n], arguments, jsonl)
        elif self.out_format == 'json':
            out_text, ids = outputs.print_json(results[n], arguments, jsonl)
        elif self.out_format == 'rasc':
            out_text_cab, ids = outputs.print_cabocha(results[n], arguments, jsonl, idx_hash[n], self.init_id,
                                                      rasc=True)
            out_text_json, ids = outputs.print_json(results[n], arguments, jsonl, rasc=True)
            out_text_pretty, ids = outputs.print_pretty(results[n], arguments, jsonl)
            out_json = json.dumps({
                'cabocha': out_text_cab,
                'json': out_text_json,
                'pretty': out_text_pretty
            }, ensure_ascii=False)
            out_text = '{}\nEOS'.format(out_json)
        elif self.out_format == 'conll':
            out_text, ids = outputs.print_conll(results[n], arguments, jsonl, idx_hash[n], self.init_id)
        else:
            raise NotImplementedError
        self.init_id += ids
        print(out_text)
