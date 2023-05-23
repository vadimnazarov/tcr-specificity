import torch
from torch.nn import TripletMarginLoss, CrossEntropyLoss, BCELoss
from torch.nn.functional import pairwise_distance
import numpy as np

from rune.model import *
from .fc import *


class AutoencoderTrainer(FCDistanceTrainer):
    
    def __init__(self):
        super(AutoencoderTrainer, self).__init__()
            
    
    def init(self, input_size, model_params, lr, device):
        self.learning_rate = lr
        self.device = device
        
        self.model = RuneAutoencoder(input_size=input_size, **model_params).to(self.device)

        self.criterion = CrossEntropyLoss()
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        self.train_metric = {"loss":[]}
        self.valid_metric = {"loss":[]}
        
    
    def run_epoch(self, dl_pos, dl_neg, train_mode=True):
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        batch_metric = {key:[] for key in self.train_metric}

        for i, (batch_neg, batch_pos) in enumerate(zip(dl_neg, dl_pos)):
            self.optimizer.zero_grad()

            vec_neg, len_neg = batch_neg[0].to(self.device), batch_neg[1].view(-1)
            vec_pos, len_pos = batch_pos[0].to(self.device), batch_pos[1].view(-1)
            
            out_neg = self.model(vec_neg, len_neg)
            out_pos = self.model(vec_pos, len_pos)
            
            print(out_pos)
            
            loss_pos = self.criterion(out_neg, vec_neg)
            loss_neg = self.criterion(out_pos, vec_pos)
            loss = loss_pos + loss_neg
            
            batch_metric["loss"].append(loss.item())

            if train_mode:
                loss.backward()
                self.optimizer.step()

        for key in batch_metric:
            if train_mode:
                self.train_metric[key].append(np.mean(batch_metric[key]))
            else:
                self.valid_metric[key].append(np.mean(batch_metric[key]))