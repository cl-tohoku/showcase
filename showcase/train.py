# -*- coding: utf-8 -*-
import argparse
import os
import random
import sys
from os import path

import numpy as np
import torch
from torch import autograd, nn
from tqdm import tqdm
from arg_detector_io import *

import utils
from arg_detector_io import end2end_single_seq_instance, pretrained_word_vecs, \
    e2e_single_seq_pos_sentence_batch_with_multi_predicate
from eval import evaluate_multiclass_without_none
from models import E2EStackedBiRNNMpPoolAttentionFixedWordEmb
from utils.inputs import InputProcessor, end2end_dataset

random.seed(2016)
np.random.seed(2016)
max_sentence_length = 90
prediction_model_id = 3


def train_pred(sub_model_number,
               data_path, data_train, data_dev, batch_size,
               model, model_id,
               epoch, lr_start, lr_min):
    early_stopping_thres = 4
    early_stopping_count = 0
    best_performance = -1.0
    best_epoch = 0
    # best_weights = []
    best_state = []
    best_thres = [0.0, 0.0]
    best_lr = lr_start
    lr = lr_start
    lr_reduce_factor = 0.5
    lr_epsilon = lr_min * 1e-4

    thres_set_pred = list(map(lambda n: n / 100.0, list(range(50, 51, 1))))
    thres_set_noun = list(map(lambda n: n / 100.0, list(range(50, 51, 1))))
    thres_lists = [thres_set_pred, thres_set_noun]
    labels = ["pred", "noun", "all"]

    loss_function = nn.NLLLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_start)
    losses = []

    for ep in range(epoch):
        # batch_for_pred_detection == e2e_instance_for_pred ??? nohazu...
        batch_train = list(InputProcessor.batch_for_pred_detection(data_train, batch_size))
        batch_dev = list(InputProcessor.batch_for_pred_detection(data_dev, batch_size))
        len_train = len(batch_train)
        len_dev = len(batch_dev)

        total_loss = torch.Tensor([0])
        early_stopping_count += 1

        print(model_id, 'epoch {0}'.format(ep + 1), flush=True)

        print('Train...', flush=True)

        model.train()
        for xss, yss in tqdm(batch_train, total=len_train, mininterval=5):
            optimizer.zero_grad()
            model.zero_grad()

            if torch.cuda.is_available():
                yss = autograd.Variable(yss).cuda()
            else:
                yss = autograd.Variable(yss)

            scores = model(xss)

            loss = 0
            # print(scores, yss)
            for i in range(yss.size()[0]):
                # print(scores[i], yss[i])
                loss += loss_function(scores[i], yss[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.data.cpu()

        print("loss:", total_loss[0], "lr:", lr)
        losses.append(total_loss)
        print("", flush=True)
        print('Test...', flush=True)

        model.eval()
        thres, obj_score, num_test_batch_instance = evaluate_predicate_prediction(model, batch_dev, len_dev, labels,
                                                                                  thres_lists)
        f = obj_score * 100
        if f > best_performance:
            best_performance = f
            early_stopping_count = 0
            best_epoch = ep + 1
            best_thres = thres
            best_lr = lr
            print("save model", flush=True)
            model.save(data_path + "/model-" + model_id + ".h5")
        elif early_stopping_count >= early_stopping_thres:
            # break
            if lr > lr_min + lr_epsilon:
                new_lr = lr * lr_reduce_factor
                lr = max(new_lr, lr_min)
                print("load model: epoch{0}".format(best_epoch), flush=True)
                model.load(data_path + "/model-" + model_id + ".h5")
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                early_stopping_count = 0
            else:
                break
        print(model_id, "\tcurrent best epoch", best_epoch, "\t", best_thres, "\t", "lr:", best_lr, "\t",
              "f:", best_performance)

    print(model_id, "\tbest in epoch", best_epoch, "\t", best_thres, "\t", "lr:", best_lr, "\t",
          "f:", best_performance)


def train_arg(data_path, data_train, data_dev,
              model, model_id,
              epoch, lr_start, lr_min):
    len_train = len(data_train)
    len_dev = len(data_dev)

    early_stopping_thres = 4
    early_stopping_count = 0
    best_performance = -1.0
    best_epoch = 0
    # best_weights = []
    best_state = []
    best_thres = [0.0, 0.0]
    best_lr = lr_start
    lr = lr_start
    lr_reduce_factor = 0.5
    lr_epsilon = lr_min * 1e-4

    thres_set_ga = list(map(lambda n: n / 100.0, list(range(10, 71, 1))))
    thres_set_wo = list(map(lambda n: n / 100.0, list(range(20, 86, 1))))
    thres_set_ni = list(map(lambda n: n / 100.0, list(range(0, 61, 1))))
    thres_lists = [thres_set_ga, thres_set_wo, thres_set_ni]
    labels = ["ga", "wo", "ni", "all"]

    loss_function = nn.NLLLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_start)
    losses = []

    for ep in range(epoch):
        total_loss = torch.Tensor([0])
        early_stopping_count += 1

        print(model_id, 'epoch {0}'.format(ep + 1), flush=True)

        print('Train...', flush=True)
        random.shuffle(data_train)

        model.train()
        for xss, yss in tqdm(data_train, total=len_train, mininterval=5):
            if yss.size(1) > max_sentence_length:
                continue

            # print(yss.size())

            optimizer.zero_grad()
            model.zero_grad()

            if torch.cuda.is_available():
                yss = autograd.Variable(yss).cuda()
            else:
                yss = autograd.Variable(yss)

            scores = model(xss)

            loss = 0
            # print(scores, yss)
            for i in range(yss.size()[0]):
                # print(scores[i], yss[i])
                loss += loss_function(scores[i], yss[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.data.cpu()

        print("loss:", total_loss[0], "lr:", lr)
        losses.append(total_loss)
        print("", flush=True)
        print('Test...', flush=True)

        model.eval()
        thres, obj_score, num_test_batch_instance = evaluate_multiclass_without_none(model, data_dev, len_dev, labels,
                                                                                     thres_lists)
        f = obj_score * 100
        if f > best_performance:
            best_performance = f
            early_stopping_count = 0
            best_epoch = ep + 1
            best_thres = thres
            best_lr = lr
            print("save model", flush=True)
            model.save(data_path + "/model-" + model_id + ".h5")
        elif early_stopping_count >= early_stopping_thres:
            # break
            if lr > lr_min + lr_epsilon:
                new_lr = lr * lr_reduce_factor
                lr = max(new_lr, lr_min)
                print("load model: epoch{0}".format(best_epoch), flush=True)
                model.load(data_path + "/model-" + model_id + ".h5")
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                early_stopping_count = 0
            else:
                break
        print(model_id, "\tcurrent best epoch", best_epoch, "\t", best_thres, "\t", "lr:", best_lr, "\t",
              "f:", best_performance)

    print(model_id, "\tbest in epoch", best_epoch, "\t", best_thres, "\t", "lr:", best_lr, "\t",
          "f:", best_performance)


def set_log_file(args, tag, model_id):
    sys.stdout = utils.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = utils.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    fd = os.open(args.data_path + '/log/log-' + tag + '-' + model_id + ".txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())


def create_pretrained_model_id(args, model_name):
    # return "{0}_vh{1}_{2}_{3}_lr{4}_di{5}_dh{6}_ve{7}_vu{8}_de{9}_dw{10}_du{11}_r{12}_b{13}_size{14}_{15}_nt{16}_fp{17}".format(
    sub_model_no = "" if args.sub_model_number == -1 \
        else "_sub{0}".format(args.sub_model_number)

    if model_name == "e2e-multi-p":
        depth = args.depth
    else:
        depth = "{0}-{1}-{2}".format(args.depth, args.depth_path, args.depth_arg)

    return "{0}_vu{1}_{2}_{3}_lr{4}_du{5}_ve{6}_b{7}_size{8}{9}".format(
        model_name,
        args.vec_size_u, depth,
        args.optim, 0.0002,
        args.drop_u,
        args.vec_size_e,
        args.batch_size,
        100,
        sub_model_no
    )


def create_model_id(args, mode):
    sub_model_no = "" if args.sub_model_number == -1 \
        else "_sub{0}".format(args.sub_model_number)

    if args.model_name == 'e2e-ffatt':
        return "{0}_vh{1}_{2}_nh{3}_{4}_lr{5}_da{6}_dr{7}_size{8}{9}".format(
            args.model_name,
            args.vec_size_h, args.depth,
            args.num_head,
            args.optim, args.lr,
            args.drop_a,
            args.drop_r,
            args.data_size,
            sub_model_no
        )
    if args.model_name == 'e2e-stack-mpp-mha':
        return "{0}_vh{1}_vp{2}_{3}_nh{4}_{5}_lr{6}_du{7}_size{8}{9}".format(
            args.model_name,
            args.vec_size_u,
            args.dim_pool,
            args.depth,
            args.num_head,
            args.optim, args.lr,
            args.drop_u,
            # args.drop_a,
            # args.drop_r,
            args.data_size,
            sub_model_no
        )

    depth = ""
    if args.model_name == 'e2e-multi-ppp' or args.model_name == 'e2e-multi-ppp2' or args.model_name == 'e2e-multi-ppp3' \
            or args.model_name == 'e2e-mppp2' or args.model_name == 'e2e-mppp2r' \
            or args.model_name == 'e2e-s' or args.model_name == 'e2e-s2' \
            or args.model_name == 'e2e-s3' or args.model_name == 'e2e-s3p' or args.model_name == 'e2e-s3p2' \
            or args.model_name == 'e2e-s3-pre' \
            or args.model_name == 'e2e-s4' or args.model_name == 'e2e-s4d' or args.model_name == 'e2e-s4s' \
            or args.model_name == 'e2e-s2-sum' or args.model_name == 'e2e-s2-avg' or args.model_name == 'e2e-s2-last':
        depth = "{0}-{1}".format(args.depth, args.depth_path)
    elif args.model_name == 'e2e' or args.model_name == 'e2e-multi-p' \
            or args.model_name == 'e2e-stack' or args.model_name == 'e2e-stack-mp' \
            or args.model_name == 'e2e-stack-p' or args.model_name == 'e2e-stack-pf' \
            or args.model_name == 'e2e-stack-mpp' or args.model_name == 'e2e-stack-mppf' \
            or args.model_name == 'e2e-stack-pa' \
            or args.model_name == 'e2e-stack-mppa' or args.model_name == 'e2e-stack-mppad' \
            or args.model_name == 'e2e-stack-mpap' or args.model_name == 'e2e-stack-ap' \
            or args.model_name == 'e2e-stack-mpsa' or args.model_name == 'e2e-stack-sa' \
            or args.model_name == 'e2e-grid' \
            or args.model_name == 'e2e-mpp-r' or args.model_name == 'e2e-mpp-rd' or args.model_name == 'e2e-mpp-rb' \
            or args.model_name == 'e2e-mpp2-r' or args.model_name == 'e2e-mpp2' or args.model_name == 'e2e-mpp2-rd' \
            or args.model_name == 'e2e-mpp2-r2' \
            or args.model_name == 'e2e-mpap' or args.model_name == 'e2e-mpap2' \
            or args.model_name == 'e2e-mppa' \
            or args.model_name == 'e2e-mp-pre':
        depth = args.depth
    else:
        depth = "{0}-{1}-{2}".format(args.depth, args.depth_path, args.depth_arg)

    dim = ""
    if args.model_name == 'e2e-mpp-r' or args.model_name == 'e2e-mpp-rd' or args.model_name == 'e2e-mpp-rb' \
            or args.model_name == 'e2e-stack-p' or args.model_name == 'e2e-stack-pf' \
            or args.model_name == 'e2e-stack-mpp' or args.model_name == 'e2e-stack-mppf' \
            or args.model_name == 'e2e-stack-pa' \
            or args.model_name == 'e2e-stack-mppa' or args.model_name == 'e2e-stack-mppad' \
            or args.model_name == 'e2e-stack-mpap' or args.model_name == 'e2e-stack-ap' \
            or args.model_name == 'e2e-stack-mpsa' or args.model_name == 'e2e-stack-sa' \
            or args.model_name == 'e2e-mpp2' or args.model_name == 'e2e-mpp2-r' or args.model_name == 'e2e-mpp2-r2' \
            or args.model_name == 'e2e-mpp2-rd' \
            or args.model_name == 'e2e-mpap' or args.model_name == 'e2e-mpap2' \
            or args.model_name == 'e2e-mppa':
        dim = 'vu{0}_vp{1}'.format(args.vec_size_u, args.dim_pool)
    elif args.model_name == "e2e-ffatt":
        dim = 'vu{0}_vh{1}'.format(args.vec_size_u, args.vec_size_h)
    else:
        dim = 'vu{0}'.format(args.vec_size_u)

    return "{0}_{1}_{2}_{3}_lr{4}_du{5}_dh{6}_size{8}{9}".format(
        args.model_name,
        dim, depth,
        args.optim, args.lr,
        args.drop_u,
        args.drop_h,
        args.batch_size,
        args.data_size,
        sub_model_no
    )


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', dest='data_path', type=path.abspath,
                        help='data path')
    parser.add_argument('--division', dest='division_path', type=path.abspath,
                        help='file including indice of data division')
    parser.add_argument('--model_no', "-mn", dest='sub_model_number', type=int,
                        help='sub-model number for ensembling', default=-1)
    parser.add_argument('--size', '-s', dest='data_size', type=int,
                        default=100,
                        help='data size (%)')
    parser.add_argument('--model', '-m', dest='model_name', type=str,
                        default="path-pair-bin",
                        help='model name')
    parser.add_argument('--epoch', '-e', dest='max_epoch', type=int,
                        default=150,
                        help='max epoch')
    parser.add_argument('--vec_e', '-ve', dest='vec_size_e', type=int,
                        default=256,
                        help='word embedding size')
    parser.add_argument('--vec_h', '-vh', dest='vec_size_h', type=int,
                        default=1000,
                        help='hidden vector size')
    parser.add_argument('--vec_pe', '-vpe', dest='vec_size_pe', type=int,
                        default=64,
                        help='word embedding size in path embedding')
    parser.add_argument('--vec_u', '-vu', dest='vec_size_u', type=int,
                        default=256,
                        help='unit vector size in rnn')
    parser.add_argument('--vec_p', '-vp', dest='dim_pool', type=int,
                        default=1024,
                        help='unit vector size in rnn')
    parser.add_argument('--depth', '-dep', '-d', dest='depth', type=int,
                        default=10,
                        help='the number of hidden layer')
    parser.add_argument('--depth-path', '-dp', '-dp', dest='depth_path', type=int,
                        default=2,
                        help='the number of hidden layer')
    # parser.add_argument('--depth-arg', '-da', '-da', dest='depth_arg', type=int,
    #                     default=2,
    #                     help='the number of hidden layer')
    parser.add_argument('--optimizer', '-o', dest='optim', type=str,
                        default="adagrad",
                        help='optimizer')
    parser.add_argument('--lr', '-l', dest='lr', type=float,
                        default=0.005,
                        help='learning rate')
    parser.add_argument('--lr-min', '-lm', dest='lr_min', type=float,
                        default=0.00005,
                        help='learning rate')
    parser.add_argument('--dropout-in', '-di', dest='drop_in', type=float,
                        default=0.0,
                        help='dropout rate of input layers')
    parser.add_argument('--dropout-embedding-in', '-de', dest='drop_e', type=float,
                        default=0.1,
                        help='dropout rate of input layers')
    parser.add_argument('--dropout-w', '-dw', dest='drop_w', type=float,
                        default=0.2,
                        help='dropout rate of LSTM weight matrix')
    parser.add_argument('--dropout-u', '-du', dest='drop_u', type=float,
                        default=0.2,
                        help='dropout rate of LSTM unit')
    parser.add_argument('--dropout-h', '-dh', dest='drop_h', type=float,
                        default=0.0,
                        help='dropout rate of hidden layers')
    parser.add_argument('--dropout-residual', '-dr', dest='drop_r', type=float,
                        default=0.1,
                        help='dropout rate applied before residual connection')
    parser.add_argument('--dropout-attention', '-da', dest='drop_a', type=float,
                        default=0.2,
                        help='dropout rate of attention softmax layer')
    parser.add_argument('--num-head', '-nh', dest='num_head', type=int,
                        default=1,
                        help='the number of heads in multi-head attention layers')
    parser.add_argument('--regularize', '-r', dest='reg', type=float,
                        default=0.0,
                        help='l2 regularization parameter')
    parser.add_argument('--batch', '-b', dest='batch_size', type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--num-false-positive', '-fp', dest='num_false_positives', type=int,
                        default=1,
                        help='batch size')
    parser.add_argument('--gpu', '-g', dest='gpu', type=int,
                        default='-1',
                        help='GPU ID for execution')
    parser.add_argument('--objective', '-obj', dest='objective', type=str,
                        default='f',
                        help='objective for early stopping')
    parser.add_argument('--path-model', '-mpath', dest='path_model', type=str,
                        default='',
                        help='pretrained path model')
    parser.add_argument('--pair-model', '-mpair', dest='pair_model', type=str,
                        default='',
                        help='pretrained pred-arg pair model')
    parser.add_argument('--with-thres', '-wt', dest='no_thres', action='store_false',
                        help='with thresholds')
    parser.add_argument('--no-thres', '-nt', dest='no_thres', action='store_true',
                        help='without thresholds')
    parser.add_argument('--with-none', '-wn', dest='with_none', action='store_true',
                        help='with "none" argument candidate')
    parser.add_argument('--init', '-init', dest='init', type=str,
                        default='glorot_uniform',
                        help='initialization method of weight vector')
    parser.set_defaults(with_none=False)
    parser.set_defaults(no_thres=False)

    return parser


def run2():
    parser = create_arg_parser()
    args = parser.parse_args()
    model_id = create_model_id(args, 2)
    print(model_id)

    # gpus = get_available_gpu(4000)[0]
    # gpu_id = int(gpus[0]) if args.gpu == -1 else args.gpu
    # print("gpus:", gpus, gpu_id)

    # gpu_id = 0
    gpu_id = args.gpu
    print("gpu:", gpu_id)

    torch.manual_seed(args.sub_model_number)
    set_log_file(args, "train", model_id)

    data_train = end2end_dataset(args.data_path + "/instances-train.txt",
                                 args.data_size)

    data_dev = end2end_dataset(args.data_path + "/instances-dev.txt",
                               args.data_size)

    evaluator = evaluate_multiclass_without_none
    model = None
    single_batch_gen = None

    if args.model_name == 'e2e-pos-mppa':
        data_train = list(
            end2end_single_seq_instance(data_train, e2e_single_seq_pos_sentence_batch_with_multi_predicate))
        data_dev = list(end2end_single_seq_instance(data_dev, e2e_single_seq_pos_sentence_batch_with_multi_predicate))
        word_embedding_matrix = pretrained_word_vecs(args.data_path, "/wordIndex.txt", args.vec_size_e)
        model = E2EStackedBiRNNMpPoolAttentionFixedWordEmb(args.vec_size_u, args.dim_pool,
                                                           args.depth, 4,
                                                           word_embedding_matrix,
                                                           args.drop_u)

    if torch.cuda.is_available():
        model = model.cuda()
        with torch.cuda.device(gpu_id):
            train_arg(args.data_path, data_train, data_dev,
                      model, model_id,
                      args.max_epoch, args.lr, args.lr / 20)
    else:
        train_arg(args.data_path, data_train, data_dev,
                  model, model_id,
                  args.max_epoch, args.lr, args.lr / 20)


if __name__ == '__main__':
    run2()
