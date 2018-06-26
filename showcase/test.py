# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import logzero
from logzero import logger

from showcase import constants
from showcase.models import ModelAPI
from showcase.utils.subfuncs import read_stdin


def parse(args):
    if not args.debug:
        logzero.loglevel(logging.INFO)

    if args.config is not None and os.path.isfile(args.config):
        logger.debug('Use custom config file at: [{}]'.format(args.config))
        constants.CONFIG_PATH = args.config

    model_api = ModelAPI(args.input_format, args.output_format, args.ensemble, args.debug, args.gpu)
    model_api.set_log_file()
    model_api.prepare_model()

    file_available = os.path.isfile(args.input_file)
    if file_available:
        logger.debug('Read file from argument: {}'.format(args.input_file))
        fi = json.load(open(args.input_file, 'r'))
    else:
        logger.debug('Read file from standard input')
        fi = read_stdin()

    model_api.infer(fi)


def setup(args):
    config_file = str(Path(__file__).absolute().parents[0].joinpath('data/config.json'))
    showcase_root = Path(os.environ.get('HOME')).joinpath('.config/showcase')
    if not showcase_root.exists():
        os.makedirs(str(showcase_root))
    logger.info('Copy vanilla config.json to [{}]'.format(showcase_root))
    shutil.copy(config_file, str(showcase_root))


def main():
    parser = argparse.ArgumentParser(description='Showcase: Japanese PAS Analyzer')
    parser.set_defaults(func=parse)
    parser.add_argument('--input_format', '-i', default='raw', choices=['raw', 'wakati'], type=str, help='input format')
    parser.add_argument('--output_format', '-o', default='cabocha',
                        choices=['cabocha', 'pretty', 'json', 'rasc', 'conll'], type=str, help='output format')
    parser.add_argument('--ensemble', action='store_true', help='10 model ensemble')
    parser.add_argument('--debug', action='store_true', help='if true, allow verbose output')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (Negative value: CPU)')
    parser.add_argument('--input_file', '-f', type=os.path.abspath, default='', help='input file (json format)')
    parser.add_argument('--config', '-c', type=os.path.abspath, help='custom configuration file (json format)')
    subparsers = parser.add_subparsers(help='sub-command help', title='subcommands')
    setup_parser = subparsers.add_parser('setup', help='setup config file')
    setup_parser.set_defaults(func=setup)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
