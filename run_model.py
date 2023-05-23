import time
import argparse
import uuid

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from rune.preprocessing import SeqProcessor
from rune.trainer import *

    
def train_model(params):
#     df = pd.read_csv('data/train.data'); label = 'A0301_KLGGALQAK_IE-1_CMV_binder'
#     df = pd.read_csv('data/train.dkm.data')
#     df = pd.read_csv('data/train_multilabel.data'); label = 'KLGGALQAK-IE-1-CMV'
#     df = pd.read_csv('data/train_multilabel.data'); label = 'NLVPMVATV-pp65-CMV'
#     df = pd.read_csv("data/train_human_antigens.data"); label = "B7-TPRVTGGGAM"
#     df = pd.read_csv('data/train_multilabel_ab.data')
#     df = pd.read_csv('data/train_t1d.data'); label = 'T1D'
    df = pd.read_csv('data/train_ms.data'); label = 'MS'
#     df = pd.read_csv('data/train_ra.data'); label = 'RA'

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = "cuda"
    else:
        device = "cpu"

    column = "TRB"
    df = df.sort_values(label, ascending=False).reset_index(drop=True)
    df = df.loc[~df.duplicated([column]), :].reset_index(drop=True)
    
    #
    # Get the data
    #
    processor = SeqProcessor(df[column].values, df[label].values, 'atchley', min_len=9, max_len=19)
    ds_neg, ds_pos = processor.get_neg_pos(max_n=params.pop("max_n"), seed=42)

    valid_size = min(len(ds_neg), len(ds_pos)) // 5

    pos_train = TensorDataset(*ds_pos[:-valid_size])
    print("The length of train positive dataset is {}".format(len(pos_train)))
    print(pos_train.tensors[0].shape)
    neg_train = TensorDataset(*ds_neg[:-valid_size])
    print("The length of train negative dataset is {}".format(len(neg_train)))
    print(neg_train.tensors[0].shape)
    
    pos_val = TensorDataset(*ds_pos[-valid_size:-valid_size//2])
    neg_val = TensorDataset(*ds_neg[-valid_size:-valid_size//2])
    
    print("The length of valid positive dataset is {}".format(len(pos_val)))
    print(pos_val.tensors[0].shape)
    
    print("The length of valid negative dataset is {}".format(len(neg_val)))
    print(neg_val.tensors[0].shape)
    
    pos_test = TensorDataset(*ds_pos[-valid_size//2:])
    neg_test = TensorDataset(*ds_neg[-valid_size//2:])
    
    print("The length of testing positive dataset is {}".format(len(pos_test)))
    print(pos_test.tensors[0].shape)
    
    print("The length of testing negative dataset is {}".format(len(neg_test)))
    print(neg_test.tensors[0].shape)

    b_size = params.pop("batch", 64)
    
    if len(ds_neg) > len(ds_pos):
        sampler_neg = RandomSampler(neg_train, replacement=False, num_samples=None)
        sampler_pos = RandomSampler(pos_train, replacement=True,  num_samples=len(neg_train))
    elif len(ds_neg) < len(ds_pos):
        sampler_neg = RandomSampler(neg_train, replacement=True,  num_samples=len(pos_train))
        sampler_pos = RandomSampler(pos_train, replacement=False, num_samples=None)
    else:
        sampler_neg = RandomSampler(neg_train, replacement=False, num_samples=None)
        sampler_pos = RandomSampler(pos_train, replacement=False, num_samples=None)

    dl_pos_train = DataLoader(pos_train, sampler=sampler_pos, batch_size=b_size, pin_memory=True, drop_last=True)
    dl_neg_train = DataLoader(neg_train, sampler=sampler_neg, batch_size=b_size, pin_memory=True, drop_last=True)
    dl_pos_val = DataLoader(pos_val, batch_size=512, shuffle=False, pin_memory=True)
    dl_neg_val = DataLoader(neg_val, batch_size=512, shuffle=False, pin_memory=True)
    dl_pos_test = DataLoader(pos_test, batch_size=512, shuffle=False, pin_memory=True)
    dl_neg_test = DataLoader(neg_test, batch_size=512, shuffle=False, pin_memory=True)
    
    #
    # ToDo: make a check in the argument parser
    #
    assert params["mode"] in ["cosine", "fc", "triplet", "ae", "clf"]
    if params["mode"] == "cosine":
        trainer = CosineEmbeddingTrainer()
    elif params["mode"] == "fc":
        trainer = FCDistanceTrainer()
    elif params["mode"] == "triplet":
        trainer = TripletManhattanTrainer()
    elif params["mode"] == "ae":
        trainer = AutoencoderTrainer()
    elif params["mode"] == "clf":
        trainer = ClassifierTrainer()
    params.pop("mode")
    
    #
    # Initalise the trainer
    # 
    epochs = params.pop("epochs", 100)
    trainer.init(len(processor.encoding_dict["A"]), 2, params, params.pop("lrate", 0.01), device)
    print(trainer.model)
    
    print("Choosing the best learning rate...")
    best_lr_value, min_loss = trainer.choose_learning_rate(dl_pos_train, dl_neg_train)
    print("Best learning rate is {0:3f} with the loss value of {1}".format(best_lr_value, min_loss))
    
#     model_id = str(uuid.uuid4())[:4]
#     with open("results/" + model_id + ".model.txt", "w") as f:
#         print(trainer.model, file=f)
#     print(model_id)
#     0/0
    
    printm = params.pop("printm")
    start = time.time()
    n_rows = 0
    for epoch in range(epochs):
        trainer.run_epoch(dl_pos_train, dl_neg_train)
        trainer.run_epoch(dl_pos_val, dl_neg_val, False)
        trainer.run_epoch(dl_pos_test, dl_neg_test, False, True)
        
        if printm:
            start, n_rows = print_table(trainer, epoch, epochs, start, n_rows)
            
def print_table(trainer, epoch, epochs, start, n_rows):
    if n_rows % 10 == 0:
        trainer.print_header()
        n_rows += 1
        
    if (epoch == 0) or ((epoch+1) % 2 == 0) or (epoch == epochs-1):
        end = time.time()
        trainer.print_metrics(epoch, end-start)
        n_rows += 1
        start = time.time()
    
    return start, n_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', default=64, type=int, help='Batch size')
    parser.add_argument('-l', '--lrate', default=0.01, type=float, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', default=20, type=int, help='Total epochs')
    parser.add_argument('-f', '--filters', default=16, type=int, dest='n_filters', help='Number of filters')
    parser.add_argument('-c', '--cnn', default=1, type=int, help='Number of CNN layers')
    parser.add_argument('-hd', '--hidden', default=16, type=int, dest='hidden_size_gru', help='Hidden size GRU')
    parser.add_argument('-dr', '--dropout_rnn', default=0.0, type=float, help='GRU dropout')
    parser.add_argument('-dc', '--dropout_cnn', default=0.0, type=float, help='CNN dropout')
    parser.add_argument('-nl', '--nlayers', default=2, type=int, dest='n_layers', help='Number of GRU layers')
    parser.add_argument('-fl', '--fclayers', default=2, type=int, dest='fc_layers', help='Number of fully connected layers for distance (used in mode=="fc" only)')
    parser.add_argument('-m', '--mode', default="fc", type=str, help='Which trainer mode to use: cosine embeddings ("cosine") or fully-connected-based distance ("fc")')
    parser.add_argument('-x', '--maxn', default=0, type=int, dest='max_n', help='Maximum number of samples for train and valid. Pass 0 (zero) to use all samples.')
    parser.add_argument('-p', '--print', dest='printm', default=False, action='store_true', help='If true, prints metrics')
    args = parser.parse_args()

    params = vars(args)

    train_model(params)

if __name__ == "__main__":
    main()
