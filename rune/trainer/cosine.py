import torch
from torch.nn import TripletMarginLoss, CrossEntropyLoss, BCELoss
from torch.nn.functional import pairwise_distance
import numpy as np
from rune.model import *


class CosineEmbeddingTrainer:
    
    """
    Trainer for cosine embedding-based model.
    sequence pair -> encoder -> cosine embedding distance
    """
    
    def __init__(self):
        pass
    
    
    def init(self, model_params, lr, device, **kwargs):
        self.learning_rate = lr
        self.device = device
        
        self.model = RuneEncoder(21, **model_params).to(self.device)
        self.criterion = torch.nn.CosineEmbeddingLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        self.train_loss = {"loss":[], "loss_pos":[], "loss_neg":[], "loss_mixed":[]}
        self.valid_loss = {"loss":[], "loss_pos":[], "loss_neg":[], "loss_mixed":[]}
        
        print("{:<20} {:<20} {:<20} {:<20}".format("epoch", "train_loss", "valid_loss", "lapse"))
        
    
    def run_epoch(self, dl_pos, dl_neg, train_mode=True):
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        batch_loss = {"loss":[], "loss_pos":[], "loss_neg":[], "loss_mixed":[]}

        for i, (batch_neg, batch_pos) in enumerate(zip(dl_pos, dl_neg)):
            self.optimizer.zero_grad()

            vec_neg, len_neg, y_neg = batch_neg
            vec_pos, len_pos, y_pos = batch_pos
            out_neg = self.model(vec_neg.to(self.device), len_neg.view(-1))
            out_pos = self.model(vec_pos.to(self.device), len_pos.view(-1))

            ind = len(out_neg) // 2
            out_neg_top = out_neg[:ind]
            out_neg_bot = out_neg[ind:]
            out_pos_top = out_pos[:ind]
            out_pos_bot = out_pos[ind:]

            y_true = torch.tensor([1] * ind, dtype=torch.float).to(self.device)
            y_false = torch.tensor([-1] * ind, dtype=torch.float).to(self.device)

            loss_neg = self.criterion(out_neg_top, out_neg_bot, y_true)
            loss_pos = self.criterion(out_pos_top, out_pos_bot, y_true)
            loss_mixed = self.criterion(out_neg_top, out_pos_top, y_false) + \
                         self.criterion(out_neg_bot, out_pos_bot, y_false)
            loss = loss_neg + loss_pos + loss_mixed

            batch_loss["loss"].append(loss.item())
            batch_loss["loss_pos"].append(loss_pos.item())
            batch_loss["loss_neg"].append(loss_neg.item())
            batch_loss["loss_mixed"].append(loss_mixed.item())

            if train_mode:
                loss.backward()
                self.optimizer.step()

        for key in batch_loss:
            if train_mode:
                self.train_loss[key].append(np.mean(batch_loss[key]))
            else:
                self.valid_loss[key].append(np.mean(batch_loss[key]))
    
    
    def print_loss(self, epoch, time_interval):
        print("{:<20} {:<20.5} {:<20.5} {:<20.5}".format(epoch+1, self.train_loss["loss"][epoch],
                                                         self.valid_loss["loss"][epoch], time_interval))