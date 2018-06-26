import io
import json
import random
from collections import defaultdict

import numpy as np
import torch
from itertools import chain, groupby
from tqdm import tqdm

# sys.setdefaultencoding('utf8')

# word_lexicon_size = 29172
# word_lexicon_size = 29336
word_lexicon_size = 29571
# pa_path_lexicon_size = 29313
pa_path_lexicon_size = 29336
pp_path_lexicon_size = 11116
lex_path_max_len = 30  # 5 lemmas, 5 POS, 4 arrows
naive_path_max_len = 10
max_determined_args = 15


def end2end_dataset(data_path, data_rate):
    data = open(data_path).readlines()
    len_data = len(data)
    data = data[:int(len_data * data_rate / 100)]
    return [json.loads(line.strip()) for line in tqdm(data, mininterval=5) if line.strip()]


def instance_par_predicate(data):
    for [[tss, pss], yss] in data:
        for i in range(yss.size(0)):
            yield [[torch.stack([tss[i]]), torch.stack([pss[i]])], torch.stack([yss[i]])]


def end2end_single_seq_instance(data, batch_generator):
    for sentence in data:
        if sentence["pas"]:
            instances = list(batch_generator(sentence))
            xss, yss = zip(*instances)
            yield [list(map(lambda e: torch.stack(e), zip(*xss))), torch.stack(yss)]
            # yield [[torch.stack(tss), torch.stack(pss)], torch.stack(yss)]


def end2end_single_seq_instance_with_prev_sentence(data, batch_generator):
    for sentence in data:
        if sentence["curr"]["pas"]:
            instances = list(batch_generator(sentence))
            xss, yss = zip(*instances)
            yield [list(map(lambda e: torch.stack(e), zip(*xss))), torch.stack(yss)]
            # yield [[torch.stack(tss), torch.stack(pss)], torch.stack(yss)]


def to_half_input(instances):
    for instance in instances:
        [[ws, ps, ts], ys] = instance
        if torch.cuda.is_available():
            yield [[ws.cuda(), ps.cuda().half(), ts.cuda().half()], ys.cuda()]
        else:
            yield instance


def e2e_single_seq_sentence_batch_with_multi_predicate(instance):
    pas_seq = instance["pas"]
    tokens = instance["tokens"]
    predicates = list(map(lambda pas: int(pas["p_id"]), pas_seq))
    for pas in pas_seq:
        p_id = int(pas["p_id"])
        ys = torch.LongTensor([int(a) for a in pas["args"]])
        ws = torch.LongTensor([int(w) for w in tokens])
        ps = torch.Tensor([[1.0] if i in predicates else [0.0] for i in range(len(tokens))])
        ts = torch.Tensor([[1.0] if i == p_id else [0.0] for i in range(len(tokens))])
        yield [[ws, ps, ts], ys]


def e2e_single_seq_pos_sentence_batch_with_multi_predicate(instance):
    pas_seq = instance["pas"]
    tokens = instance["tokens"]
    pos = instance["pos"]
    predicates = list(map(lambda pas: int(pas["p_id"]), pas_seq))
    p_types = list(map(lambda pas: int(pas["p_type"]), pas_seq))
    for pas in pas_seq:
        p_id = int(pas["p_id"])
        p_type = int(pas["p_type"])
        ys = torch.LongTensor([int(a) for a in pas["args"]])
        ws = torch.LongTensor([int(w) for w in tokens])
        ss = torch.LongTensor([int(p) for p in pos])
        ps = torch.LongTensor([p_types[predicates.index(i)] if i in predicates else 0 for i in range(len(tokens))])
        ts = torch.LongTensor([p_type if i == p_id else 0 for i in range(len(tokens))])
        yield [[ws, ss, ps, ts], ys]


def e2e_single_seq_sentence_batch_ap(instance):
    pas_seq = instance["pas"]
    tokens = instance["tokens"]
    for pas in pas_seq:
        p_id = int(pas["p_id"])
        ys = torch.LongTensor([int(a) for a in pas["args"]])
        ts = torch.LongTensor([int(t) for t in tokens])
        # pathes = torch.Tensor([[1.0] if i == p_id else [0.0] for i in range(len(tokens))])
        # pathes = [[[1.0] if i == j else [0.0] for i in range(len(tokens))] for i in range(len(tokens))]
        ps = torch.Tensor([[1.0] if i == p_id else [0.0] for i in range(len(tokens))])
        yield [[ts, ps], ys]


def e2e_single_seq_sentence_batch(instance):
    pas_seq = instance["pas"]
    tokens = instance["tokens"]
    for pas in pas_seq:
        p_id = int(pas["p_id"])
        ys = torch.LongTensor([int(a) for a in pas["args"]])
        ts = torch.LongTensor([int(t) for t in tokens])
        ps = torch.Tensor([[1.0] if i == p_id else [0.0] for i in range(len(tokens))])
        yield [[ts, ps], ys]


def e2e_single_seq_pos_sentence_batch(instance):
    pas_seq = instance["pas"]
    tokens = instance["tokens"]
    pos = instance["pos"]
    for pas in pas_seq:
        p_id = int(pas["p_id"])
        ys = torch.LongTensor([int(a) for a in pas["args"]])
        ts = torch.LongTensor([int(t) for t in tokens])
        ss = torch.LongTensor([int(p) for p in pos])
        ps = torch.Tensor([[1.0] if i == p_id else [0.0] for i in range(len(tokens))])
        yield [[ts, ss, ps], ys]


def e2e_instance_for_pred(instances, batch_size):
    random.shuffle(instances)
    instances = sorted(instances, key=lambda instance: len(instance["tokens"]))

    batches = []
    for (k, g) in groupby(instances, key=lambda instance: len(instance["tokens"])):
        batches += e2e_batch_grouped_by_sentence_length_for_pred(list(g), batch_size)

    random.shuffle(batches)
    return batches


def e2e_batch_grouped_by_sentence_length_for_pred(instances, batch_size):
    remaining = 0 if len(instances) % batch_size == 0 else 1

    num_batch = len(instances) // batch_size + remaining
    # print(len(instances), len(instances) / batch_size, remaining)

    for i in range(num_batch):
        start = i * batch_size
        b = instances[start:start + batch_size]
        xs = []
        ys = []
        for instance in b:
            pas_seq = instance["pas"]
            tokens = torch.LongTensor([int(t) for t in instance["tokens"]])
            pos = torch.LongTensor([int(p) for p in instance["pos"]])
            ps = torch.LongTensor([0 for i in range(len(tokens))])

            for pas in pas_seq:
                p_id = int(pas["p_id"])
                p_type = int(pas["p_type"])
                ps[p_id] = p_type
            xs.append([tokens, pos])
            ys.append(ps)
        yield [list(map(lambda e: torch.stack(e), zip(*xs))), torch.stack(ys)]


def batch_generator(xys, max_feature, batch_size, train_num_batch, single_batch):
    for i in range(train_num_batch):
        start = i * batch_size
        xs, ys = zip(*xys[start:start + batch_size])
        yss = np.array(ys).T
        yield single_batch(xs, [yss[0], yss[1], yss[2]], max_feature)


def e2e_sentence_batch_with_prev_sentence(instance):
    pas_seq = instance["curr"]["pas"]
    prev_tokens = instance["prev"]["tokens"]
    tokens = instance["curr"]["tokens"]
    predicates = list(map(lambda pas: int(pas["p_id"]), pas_seq))
    for pas in pas_seq:
        p_id = int(pas["p_id"])
        ys = torch.LongTensor([int(a) for a in pas["args"]])
        ws = torch.LongTensor([int(w) for w in tokens])
        p_ws = torch.LongTensor([int(t) for t in prev_tokens]) if prev_tokens else torch.LongTensor([0])
        ps = torch.Tensor([[1.0] if i in predicates else [0.0] for i in range(len(tokens))])
        ts = torch.Tensor([[1.0] if i == p_id else [0.0] for i in range(len(tokens))])
        yield [[p_ws, ws, ps, ts], ys]


def end2end_single_seq_instance_with_prev_sentences(data):
    for sentence in data:
        if sentence["curr"]["pas"]:
            prev_ss = sentence["prev"]
            pas_seq = sentence["curr"]["pas"]
            tokens = sentence["curr"]["tokens"]
            predicates = list(map(lambda pas: int(pas["p_id"]), pas_seq))

            p_wss = []
            for prev_s in prev_ss:
                prev_tokens = prev_s["tokens"]
                p_ws = torch.LongTensor([int(t) for t in prev_tokens]) if prev_tokens else torch.LongTensor([0])
                p_wss += [p_ws]

            instances = []
            for pas in pas_seq:
                p_id = int(pas["p_id"])
                ys = torch.LongTensor([int(a) for a in pas["args"]])
                ws = torch.LongTensor([int(w) for w in tokens])
                ps = torch.Tensor([[1.0] if i in predicates else [0.0] for i in range(len(tokens))])
                ts = torch.Tensor([[1.0] if i == p_id else [0.0] for i in range(len(tokens))])
                instances += [[ws, ps, ts], ys]

            xss, yss = zip(*instances)
            yield [p_wss, list(map(lambda e: torch.stack(e), zip(*xss))), torch.stack(yss)]


def e2e_single_seq_batch_generator(instances, num_instances, batch_size):
    for i in range(num_instances):
        start = i * batch_size
        xss, yss = zip(*instances[start:start + batch_size])
        tss, pss = zip(*xss)
        print("tss", torch.stack(tss))
        print("pss", pss)
        print("yss", yss)
        yield [torch.stack(tss), torch.stack(pss)], torch.stack(yss)


def dataset(data_path, path_max_len, parse_line):
    print('Read data...', data_path, flush=True)
    data, max_feature = read_data(data_path, path_max_len, parse_line)
    size = len(data)
    print("batch size", size, flush=True)
    return data, size, max_feature


def sentence_wise_dataset(data_path, path_max_len, parse_line, data_rate):
    print('Read data...', data_path, flush=True)
    data, max_feature = predicatewise_sentence_batches(data_path, path_max_len, parse_line, data_rate)
    size = len(data)
    print("batch size", size, flush=True)
    return data, size, int(max_feature)


def read_data_division(division_path):
    return [[np.array(train), np.array(test)] for train, test in json.load(open(division_path))]


def prediate_wise_data_2_token_wise_data(data):
    return list(chain.from_iterable(chain.from_iterable(data)))
    # return data.flatten().flatten()


def read_data(data_path, path_max_len, parse_line):
    data = tqdm(open(data_path).readlines(), mininterval=5)
    xys = []
    max_feature = 0
    xys_append = xys.append

    for line in data:
        if line == "" or line.startswith("SOD") or line.startswith("SOS") or line.startswith("SOP") or \
                line.startswith("EOD") or line.startswith("EOS") or line.startswith("EOP"):
            continue
        else:
            x, y, max_feature = parse_line(line, max_feature, path_max_len)
            xys_append([x, y])
    data.close()
    return xys, max_feature + 1


def batch_generator(xys, max_feature, batch_size, train_num_batch, single_batch):
    for i in range(train_num_batch):
        start = i * batch_size
        xs, ys = zip(*xys[start:start + batch_size])
        yss = np.array(ys).T
        yield single_batch(xs, [yss[0], yss[1], yss[2]], max_feature)


def batch_generator_multiclass(xys, max_feature, batch_size, train_num_batch, single_batch):
    for i in range(train_num_batch):
        start = i * batch_size
        xs, ys = zip(*xys[start:start + batch_size])
        yield single_batch(xs, np.array(ys), max_feature)


def create_batch(xys, max_feature, single_batch):
    xs, ys = zip(*xys)
    yss = np.array(ys).T  # ys for each label

    return single_batch(xs, yss, max_feature)


def create_predicate_batch_for_local_model(xys, max_feature):
    xs, ys = zip(*xys)
    yss = np.array(ys).T  # ys for each label
    return single_batch_path_pa_bin(xs, yss, max_feature)


def create_predicate_batch_for_global_model(xs_global, yss, pred_arg_reprs, max_feature, determined_args, pp_path):
    return single_batch_global_papa(xs_global, yss, max_feature, determined_args, pp_path)


def create_predicate_batch(xys, max_feature, determined_args, pp_path):
    xs, ys = zip(*xys)
    yss = np.array(ys).T  # ys for each label
    return single_batch_global_path_pa_bin(xs, yss, max_feature, determined_args, pp_path)


def create_batch_multiclass(xys, max_feature, single_batch):
    xs, ys = zip(*xys)

    return single_batch(xs, np.array(ys), max_feature)


def predicatewise_sentence_batches(data_path, path_max_len, parse_line, data_rate):
    # data = tqdm(io.open(data_path, "r", encoding="utf-8").readlines(), mininterval=5)
    data = io.open(data_path, "r", encoding="utf-8").readlines()
    count_doc = sum(line.startswith("EOD") for line in data)
    count_doc = int(count_doc * data_rate / 100)
    bs = []
    sb = []
    pb = []
    max_feature = 0
    s_id = 0
    p_id = 0

    doc_count = 0

    bs_append = bs.append
    sb_append = sb.append
    pb_append = pb.append

    for line in tqdm(data, mininterval=5):
        if line == "" or line.startswith("SOD"):
            continue
        elif line.startswith("EOD"):
            doc_count += 1
            if doc_count > count_doc:
                break
            continue
        elif line.startswith("SOP"):
            pb = []
            pb_append = pb.append
            continue
        elif line.startswith("EOP"):
            if pb:
                sb_append(pb)
            continue
        elif line.startswith("SOS"):
            sb = []
            sb_append = sb.append
            continue
        elif line.startswith("EOS"):
            if sb:
                bs_append(sb)
            continue
        else:
            x, y, max_feature = parse_line(line, max_feature, path_max_len)
            pb_append([x, y])
    return bs, max_feature + 1


def parse_data_line_roth(line, max_feature, path_max_len):
    labels, pred_arg, syn_path, binary_features = line.split('\t')[0:4]

    # label_arr = label_vec(labels)

    pred_arg = np.array([int(idx) for idx in pred_arg.split(":")])
    syn_path = np.array([int(idx) for idx in syn_path.split("|")][:-1:])
    syn_path = np.pad(syn_path, (path_max_len - len(syn_path), 0), 'constant', constant_values=0)

    fs = np.array([int(idx) for idx in binary_features.split(' ')], dtype='int')

    max_feature = max([max(fs), max_feature])

    return np.array([pred_arg, syn_path, fs]), label(labels), max_feature


def parse_data_line(line, max_feature, path_max_len):
    labels, pred_arg, syn_path, binary_features = line.split('\t')[0:4]

    label_arr = label_vec(labels)

    pred_arg = [int(idx) for idx in pred_arg.split(":")]
    syn_path = np.array([int(idx) for idx in syn_path.split("|")])
    syn_path = np.pad(syn_path, (path_max_len - len(syn_path), 0), 'constant', constant_values=0)

    fs = np.array([int(idx) for idx in binary_features.split(' ')], dtype='int')

    max_feature = max([max(fs), max_feature])

    return np.array([pred_arg, syn_path, fs]), label_arr, max_feature


def read_preds_path_data(data_path):
    pp_path = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    data = tqdm(open(data_path).readlines(), mininterval=5)

    for line in data:
        sentence_id, pred1_posi, pred2_posi, syn_path, pred1_idx, pred2_idx = line.strip().split('\t')
        syn_path = np.array([int(idx) for idx in syn_path.split("|")])
        syn_path = np.pad(syn_path, (lex_path_max_len - len(syn_path), 0), 'constant', constant_values=0)
        pp_path[int(sentence_id)][int(pred1_posi)][int(pred2_posi)] = syn_path
    return pp_path


def label(labels):
    ls = np.array([int(idx) for idx in labels.split(",")], dtype='int')
    return ls[0]


def label_vec(labels):
    label_arr = np.zeros(4, dtype='int')
    ls = np.array([int(idx) for idx in labels.split(",")], dtype='int')
    label_arr[ls] = 1
    return label_arr


def feature_vec(binary_features, max_feature: int):
    fs = np.zeros(max_feature)
    fs[binary_features] = 1
    return torch.from_numpy(fs).float()


def pretrained_word_vecs(data_path, index_file_name, size_e):
    wIdx = {}
    lexicon_size = 0
    wIdxData = open(data_path + index_file_name)
    for line in wIdxData:
        w, idx = line.rstrip().split("\t")
        idx = int(idx)
        wIdx[w] = idx
        if lexicon_size < idx:
            lexicon_size = idx + 1
    wIdxData.close()

    vocab_size = 0
    # wVecData = open("{0}/word_vec_{1}.txt".format(data_path, size_e))
    wVecData = open("{0}/lemma-oov_vec-{1}-jawiki20160901-filtered.txt".format(data_path, size_e))
    matrix = np.random.uniform(-0.05, 0.05, (lexicon_size, size_e))
    for line in wVecData:
        values = line.rstrip().split(" ")
        word = values[0]
        # print(word)

        if word in wIdx:
            matrix[wIdx[word]] = np.asarray(values[1:], dtype='float32')
            vocab_size += 1
    wVecData.close()

    matrix[0] = np.zeros(size_e)

    print("vocab size: ", vocab_size)
    return torch.from_numpy(matrix).float()


def pretrained_word_vecs_tmp(data_path, index_file_name, size_e, lexicon_size):
    wIdx = {}
    wIdxData = open(data_path + index_file_name)
    for line in wIdxData:
        w, idx = line.rstrip().split("\t")
        wIdx[w] = int(idx)
    wIdxData.close()

    vocab_size = 0
    wVecData = open("{0}/word_vec_{1}.txt".format(data_path, size_e))
    matrix = torch.Tensor(lexicon_size, size_e)
    for line in wVecData:
        values = line.rstrip().split(" ")
        word = values[0]

        if word in wIdx:
            matrix[wIdx[word]].append(map(float, values[1:]))
            vocab_size += 1
    wVecData.close()

    matrix[0] = torch.zeros(size_e)

    print("vocab size: ", vocab_size)
    return matrix


def single_batch_global_papa(xs_global, pa_repr, yss, max_feature, determined_args, pp_path):
    xs = []
    xs_append = xs.append

    for pred_arg, syn_path, fs, s_id, p1_id in xs_global:
        p1, a1 = pred_arg

        def determined_args_features(i):
            if i < len(determined_args):
                p2_id, p2, a2, casemk, pred_no = determined_args[i]
                p_id_same = 1 if p1_id == p2_id else 0
                a_same = 1 if a1 == a2 else 0
                bias = 1

                cm = [0, 0, 0]
                cm[casemk] = 1
                return [pp_path[s_id][p1_id][p2_id], [p1, a1], [p2, a2], cm, [p_id_same, a_same, bias]]
            else:
                return [np.zeros(lex_path_max_len), [0, 0], [0, 0], [0, 0, 0], [0, 0, 0]]

        pp_path_seq, pa1_seq, pa2_seq, cm_seq, papa_bin_seq = \
            zip(*[determined_args_features(i) for i in range(max_determined_args)])

        xs_append(
            [syn_path, pred_arg, feature_vec(fs, max_feature),
             np.array(pp_path_seq), np.array(pa1_seq), np.array(pa2_seq), np.array(cm_seq), np.array(papa_bin_seq)])

    syn_path, pred_arg, f_vec, pp_path_seq, pa1_seq, pa2_seq, cm_seq, papa_bin_seq = zip(*xs)

    return [np.array(syn_path), np.array(pred_arg), np.array(f_vec),
            np.array(pp_path_seq), np.array(pa1_seq), np.array(pa2_seq), np.array(cm_seq), np.array(papa_bin_seq)], \
           yss


def single_batch_global_path_pa_bin(batch_X, batch_Y, max_feature, determined_args, pp_path):
    xs_local = []
    xs_global = []
    xs_local_append = xs_local.append
    xs_global_append = xs_global.append

    for pred_arg, syn_path, fs, s_id, p1_id in batch_X:
        p1, a1 = pred_arg

        xs_local_append([syn_path, pred_arg, feature_vec(fs, max_feature)])
        xs_global_append([s_id, p1_id, p1, a1])

    syn_path, pred_arg, f_vec = zip(*xs_local)
    s_ids, p1_ids, p1s, a1s = zip(*xs_global)

    xs_local = [np.array(syn_path), np.array(pred_arg), np.array(f_vec)]
    xs_global = [np.array(s_ids), np.array(p1_ids), np.array(p1s), np.array(a1s)]

    return xs_local, xs_global, batch_Y


def single_batch_path_pa_bin(batch_X, batch_Y, max_feature):
    X = [[torch.from_numpy(syn_path).long(), pred_arg[0], pred_arg[1], feature_vec(fs, max_feature)] for
         pred_arg, syn_path, fs in batch_X]
    syn_path, pred, arg, f_vec = zip(*X)
    return [torch.stack(syn_path), torch.Tensor(pred).long(), torch.Tensor(arg).long(), torch.stack(f_vec)], batch_Y


def single_batch_path_pa(batch_X, batch_Y, max_feature):
    X = [[syn_path, pred_arg] for pred_arg, syn_path, fs in batch_X]
    syn_path, pred_arg = zip(*X)
    return [np.array(syn_path), np.array(pred_arg)], batch_Y


def single_batch_pa_bin(batch_X, batch_Y, max_feature):
    X = [[pred_arg, feature_vec(fs, max_feature)] for pred_arg, syn_path, fs in batch_X]
    pred_arg, f_vec = zip(*X)
    return [np.array(pred_arg), np.array(f_vec)], batch_Y


def single_batch_path_bin(batch_X, batch_Y, max_feature):
    X = [[syn_path, feature_vec(fs, max_feature)] for pred_arg, syn_path, fs in batch_X]
    syn_path, f_vec = zip(*X)
    return [np.array(syn_path), np.array(f_vec)], batch_Y


def single_batch_bin(batch_X, batch_Y, max_feature):
    X = [feature_vec(fs, max_feature) for pred_arg, syn_path, fs in batch_X]
    return torch.stack(X), batch_Y


def single_batch_path(batch_X, batch_Y, max_feature):
    X = [syn_path for pred_arg, syn_path, fs in batch_X]
    return np.array(X), batch_Y


def single_batch_pa(batch_X, batch_Y, max_feature):
    X = [pred_arg for pred_arg, syn_path, fs in batch_X]
    return np.array(X), batch_Y
