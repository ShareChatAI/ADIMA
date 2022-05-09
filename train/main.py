import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import h5py
import argparse
import time
import math
import logging
from sklearn import metrics
from utils import utilities, data_generator
import trainer
from wav2vec import Wav2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import pdb

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.weight.data.fill_(1.)

def train(args):
    model = Wav2Vec(args.cls_type, False, args.dropout)
    args.train_feat_wav_path =  args.feats_base_path + args.src_lang
    args.test_feat_wav_path = args.feats_base_path + args.tgt_lang
    trainer.train(args, model)

def set_seed(seed_val):
    np.random.seed(seed_val) 
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
        torch.cuda.manual_seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--cls_type', default="fc_avg", choices=["fc_avg", "fc_max", "lstm", "gru"])
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--duration', type=int, default=60000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--log_offset', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--src_lang', type=str, default='Hindi', choices=["Hindi", "Bengali", "Gujarati", "Kannada", "Malayalam", 
    "Punjabi", "Tamil", "Bhojpuri", "Odia", "Haryanvi"])
    parser.add_argument('--tgt_lang', type=str, default='Hindi', choices=["Hindi", "Bengali", "Gujarati", "Kannada", "Malayalam", 
    "Punjabi", "Tamil", "Bhojpuri", "Odia", "Haryanvi"])
    parser.add_argument('--csv_path', type=str, default='./SC_abuse_detection/')
    parser.add_argument('--feats_base_path', type=str, default='./features/wav2vec_features_xlsr/')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--shift_pct', type=float, default=0.1)    
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    args.filename = utilities.get_filename(__file__)

    set_seed(args.seed)

    args.train_csv_path = args.csv_path + f"{args.src_lang}_train.csv"
    args.test_csv_path = args.csv_path + f"{args.tgt_lang}_test.csv"

    print(args)

    args.duration_seconds = args.duration / 1000
    
    if args.mode == "train":
        train(args)
    else:
        raise Exception("Error!")
