# -*- coding: utf-8 -*-

"""
--------------------------------------------
File Name:          main
Author:             deepgray
--------------------------------------------
Description:


--------------------------------------------
Date:               18-7-2
Change Activity:

--------------------------------------------
"""

from model import *
import prep
import argparse
import logging
import os
import sys
import signal


logging.basicConfig(filename='log.txt', level=logging.DEBUG,
                    format='%(asctime)s%(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description="Run Commands")
parser.add_argument('-t', '--train', default=True, type=bool, help="Training mode")
parser.add_argument('--data', default="./data/", type=str, help="training data path")
parser.add_argument('--test', default=True, type=bool, help="Testing mode")
parser.add_argument("--dict_size", default=1024, type=int, help="dictionary size")
parser.add_argument("--lamb", default=0.2, type=float, help="value of lambda in model")
parser.add_argument("--load_model", default="./model/model.npy", help="Model path to save")


# test_model = SparseCode((1024, 1024), 0.2, None, None)
#
# prep.train("./test_data/train/", test_model, './model/test.npy')
# prep.test("./test_data/test/", test_model, './test_data/')
#
# prep.show(test_model)


def run(args):
    lamb = args.lamb
    dict_size = args.dict_size
    model_path = args.load_model

    model = SparseCode((dict_size, dict_size), lamb, None, None)
    if os.path.exists(model_path):
        logging.info("Loading model already trained from"+model_path)
        model.load(model_path)
    if args.train:
        prep.train(args.data+"train/", model, model_path)
        logging.info("Training Finished.")
    if args.test:
        prep.test(args.data+"test/", model, args.data + "result/")
    return


def main():
    args, _ = parser.parse_known_args()

    def shutdown(sig):
        # 日志记录
        logging.warning("Program terminated.")
        sys.exit(128+sig)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    run(args)


main()
