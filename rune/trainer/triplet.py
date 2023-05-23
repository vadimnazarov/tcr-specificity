import torch
from torch.nn import TripletMarginLoss, CrossEntropyLoss, BCELoss
from torch.nn.functional import pairwise_distance
import numpy as np

from .fc import *
from rune.model import *
        

class TripletManhattanTrainer(FCDistanceTrainer):
    
    def __init__(self):
        super(TripletManhattanTrainer, self).__init__()
            
    
    def init(self, input_size, model_params, lr, device):
        self.learning_rate = lr
        self.device = device
        
        self.model = RuneEncoder(input_size=input_size, **model_params).to(self.device)
        self.margin = 2
        
#         self.dist_fun = lambda x, y: torch.exp(-pairwise_distance(x, y, p=2).add(-self.margin).clamp(min=0))
#         self.dist_fun = lambda x, y: torch.exp(-pairwise_distance(x, y, p=2))
        self.dist_fun = lambda x, y: (self.margin - pairwise_distance(x, y, p=2)).div(self.margin).clamp(min=0)

        self.criterion = TripletMarginLoss(p=2, margin=self.margin)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        
        self.train_metric = {"loss":[], "loss_pos":[], "loss_neg":[], 
                             "mixed_acc":[], "pos_acc":[], "full_acc":[]}
        self.valid_metric = {"loss":[], "loss_pos":[], "loss_neg":[], 
                             "mixed_acc":[], "pos_acc":[], "full_acc":[]}
        
    
    def run_epoch(self, dl_pos, dl_neg, train_mode=True):
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        batch_metric = {key:[] for key in self.train_metric}

        # torch.exp(-torch.abs(x - y))
        for i, (batch_neg, batch_pos) in enumerate(zip(dl_neg, dl_pos)):
            self.optimizer.zero_grad()

            vec_neg, len_neg = batch_neg[0].to(self.device), batch_neg[1].view(-1)
            vec_pos, len_pos = batch_pos[0].to(self.device), batch_pos[1].view(-1)
            
            out_neg = self.model(vec_neg, len_neg)
            out_pos = self.model(vec_pos, len_pos)

            ind = len(out_neg) // 2
            out_neg_top = out_neg[:ind]
            out_neg_bot = out_neg[ind:]
            out_pos_top = out_pos[:ind]
            out_pos_bot = out_pos[ind:]
            
            y_pred_pos = self.dist_fun(out_pos_top, out_pos_bot)
            y_pred_mix = self.dist_fun(out_neg, out_pos)
            
            loss_pos = self.criterion(out_pos_top, out_pos_bot, out_neg_top)
            loss_neg = self.criterion(out_pos_bot, out_pos_top, out_neg_bot)
            loss = loss_pos + loss_neg
            
            batch_metric["loss"].append(loss.item())
            batch_metric["loss_pos"].append(loss_pos.item())
            batch_metric["loss_neg"].append(loss_neg.item())
            batch_metric["mixed_acc"].append(1 - y_pred_mix.round()
                                                         .sum()
                                                         .div(len(y_pred_mix))
                                                         .item())
            batch_metric["pos_acc"].append(y_pred_pos.round()
                                                       .sum()
                                                       .div(len(y_pred_pos))
                                                       .item())
            batch_metric["full_acc"].append(np.mean([batch_metric["mixed_acc"][-1], batch_metric["pos_acc"][-1]]))

            if train_mode:
                loss.backward()
                self.optimizer.step()

        for key in batch_metric:
            if train_mode:
                self.train_metric[key].append(np.mean(batch_metric[key]))
            else:
                self.valid_metric[key].append(np.mean(batch_metric[key]))
                
#         self.scheduler.step()