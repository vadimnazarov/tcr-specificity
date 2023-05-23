import torch
from torch.nn import TripletMarginLoss, CrossEntropyLoss, BCELoss
from torch.nn.functional import pairwise_distance
import numpy as np
from rune.model import *
    
    
class FCDistanceTrainer():
    
    """
    Trainer for Fully Connected distance model.
    sequence pair -> encoder -> concatenation of embeddings -> FC layers -> sigmoid (prob. of similarity)
    """
    
    def __init__(self):
        pass

    
    def init(self, model_params, lr, device):
        self.learning_rate = lr
        self.device = device
        
        self.model = RuneRSP(input_size=21, **model_params).to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        self.train_metric = {"loss":[], "loss_pos":[], "loss_neg":[], "loss_mixed":[],
                             "mixed_acc":[], "neg_acc":[], "pos_acc":[]}
        self.valid_metric = {"loss":[], "loss_pos":[], "loss_neg":[], "loss_mixed":[],
                             "mixed_acc":[], "neg_acc":[], "pos_acc":[]}
        
    
    def run_epoch(self, dl_pos, dl_neg, train_mode=True):
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        batch_metric = {key:[] for key in self.train_metric}

        for i, (batch_neg, batch_pos) in enumerate(zip(dl_pos, dl_neg)):
            self.optimizer.zero_grad()

            vec_neg, len_neg = batch_neg[0].to(self.device), batch_neg[1].view(-1)
            vec_pos, len_pos = batch_pos[0].to(self.device), batch_pos[1].view(-1)
            
            y_pred_mixed = self.model(vec_neg, len_neg, vec_pos, len_pos)
            loss_mixed = self.criterion(y_pred_mixed, 
                                        torch.tensor([0] * len(vec_neg), dtype=torch.float).to(self.device).view(-1, 1))

            ind = len(vec_pos) // 2
            y_pred_pos = self.model(vec_pos[:ind], len_pos[:ind], vec_pos[ind:], len_pos[ind:])
            loss_pos = self.criterion(y_pred_pos, 
                                      torch.tensor([1] * ind, dtype=torch.float).to(self.device).view(-1, 1))
            
            ind = len(vec_neg) // 2
            y_pred_neg = self.model(vec_neg[:ind], len_neg[:ind], vec_neg[ind:], len_neg[ind:])
            loss_neg = self.criterion(y_pred_neg, 
                                      torch.tensor([1] * ind, dtype=torch.float).to(self.device).view(-1, 1))

            loss = loss_neg + loss_pos + loss_mixed

            batch_metric["loss"].append(loss.item())
            batch_metric["loss_pos"].append(loss_pos.item())
            batch_metric["loss_neg"].append(loss_neg.item())
            batch_metric["loss_mixed"].append(loss_mixed.item())
            batch_metric["mixed_acc"].append(1 - y_pred_mixed.round()
                                                         .sum()
                                                         .div(len(y_pred_mixed))
                                                         .item())
            batch_metric["neg_acc"].append(y_pred_neg.round()
                                                       .sum()
                                                       .div(len(y_pred_neg))
                                                       .item())
            batch_metric["pos_acc"].append(y_pred_pos.round()
                                                       .sum()
                                                       .div(len(y_pred_pos))
                                                       .item())

            if train_mode:
                loss.backward()
                self.optimizer.step()

        for key in batch_metric:
            if train_mode:
                self.train_metric[key].append(np.mean(batch_metric[key]))
            else:
                self.valid_metric[key].append(np.mean(batch_metric[key]))
                
    def print_header(self):
        n_cols = len(self.train_metric.keys()) + 2
        string = "{:<14} " * n_cols
        str_fmt = "mode " + string
        columns = ["epoch"]
        for key in self.train_metric.keys():
            columns.append(key)
        columns.append("lapse")
        header = str_fmt.format(*columns)
        print(header)

    def print_metrics(self, epoch, time_interval):
        n_cols = len(self.train_metric.keys()) + 1
        string = "{:<14.3} " * n_cols
        str_fmt = "train {:<14} " + string
        print(str_fmt.format(epoch+1, *[self.train_metric[key][epoch] for key in self.train_metric], time_interval))
        str_fmt = "valid {:<14} " + string
        print(str_fmt.format(epoch+1, *[self.valid_metric[key][epoch] for key in self.valid_metric], time_interval))