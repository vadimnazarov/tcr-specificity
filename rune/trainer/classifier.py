import torch
from torch.nn import TripletMarginLoss, CrossEntropyLoss, BCELoss
from torch.nn.functional import pairwise_distance
import numpy as np
from sklearn.metrics import roc_auc_score

from rune.model import *
from .fc import *


class ClassifierTrainer(FCDistanceTrainer):
    
    def __init__(self):
        super(ClassifierTrainer, self).__init__()
            
    
    def init(self, input_size, output_size, model_params, lr, device):
        self.input_size = input_size
        self.output_size = output_size
        self.model_params = model_params
        self.learning_rate = lr
        self.device = device
        
        self.model = RuneClassifier(input_size=self.input_size, output_size=self.output_size, **model_params).to(self.device)
        if self.output_size == 2:
            self.criterion = BCELoss()
        else:
            self.criterion = CrossEntropyLoss()
            
        
#         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.0001, max_lr=0.01, cycle_momentum=False)
        
        self.train_metric = {"loss":[], "loss_pos":[], "loss_neg":[], 
                             "neg_acc":[], "pos_acc":[], "full_acc":[]}
        self.valid_metric = {"loss":[], "loss_pos":[], "loss_neg":[], 
                             "neg_acc":[], "pos_acc":[], "full_acc":[]}
        
    
    def choose_learning_rate(self, dl_pos, dl_neg):        
        model = RuneClassifier(input_size=self.input_size, output_size = self.output_size, **self.model_params).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.1)
        min_loss = 100
        
        for i, (batch_neg, batch_pos) in enumerate(zip(dl_neg, dl_pos)):
            optimizer.zero_grad()

            vec_neg, len_neg = batch_neg[0].to(self.device), batch_neg[1].view(-1)
            vec_pos, len_pos = batch_pos[0].to(self.device), batch_pos[1].view(-1)
            
            y_pred_neg = model(vec_neg, len_neg).reshape((-1,))
            y_pred_pos = model(vec_pos, len_pos).reshape((-1,))
            
            loss_neg = self.criterion(y_pred_neg, torch.tensor([0.0] * len(y_pred_neg)))
            loss_pos = self.criterion(y_pred_pos, torch.tensor([1.0] * len(y_pred_pos)))
            loss = loss_pos + loss_neg
            
            lr_value = -1
            for param_group in optimizer.param_groups:
                lr_value = param_group['lr']
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_lr_value = lr_value
            
            if lr_value > 1:
                break
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        return best_lr_value, min_loss
        
    
    def run_epoch(self, dl_pos, dl_neg, train_mode=True, test_mode=False):
        """
        For 2-class classification case.
        """
        
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        batch_metric = {key:[] for key in self.train_metric}

        for i, (batch_neg, batch_pos) in enumerate(zip(dl_neg, dl_pos)):
            self.optimizer.zero_grad()

            vec_neg, len_neg = batch_neg[0].to(self.device), batch_neg[1].view(-1)
            vec_pos, len_pos = batch_pos[0].to(self.device), batch_pos[1].view(-1)
            
            y_pred_neg = self.model(vec_neg, len_neg).reshape((-1,))
            y_pred_pos = self.model(vec_pos, len_pos).reshape((-1,))
            
            loss_neg = self.criterion(y_pred_neg, torch.tensor([0.0] * len(y_pred_neg)))
            loss_pos = self.criterion(y_pred_pos, torch.tensor([1.0] * len(y_pred_pos)))
            loss = loss_pos + loss_neg
            
            batch_metric["loss"].append(loss.item())
            batch_metric["loss_pos"].append(loss_pos.item())
            batch_metric["loss_neg"].append(loss_neg.item())
            batch_metric["neg_acc"].append(1 - y_pred_neg.round()
                                                       .sum()
                                                       .div(len(y_pred_neg))
                                                       .item())
            batch_metric["pos_acc"].append(y_pred_pos.round()
                                                       .sum()
                                                       .div(len(y_pred_pos))
                                                       .item())
#             batch_metric["full_acc"].append(.5 * (batch_metric["neg_acc"][-1] + batch_metric["pos_acc"][-1]))

            neg_scores = y_pred_neg.detach().numpy().reshape((-1,1))
            pos_scores = y_pred_pos.detach().numpy().reshape((-1,1))
            batch_metric["full_acc"].append(roc_auc_score(np.vstack([np.zeros(neg_scores.shape), np.ones(pos_scores.shape)]), np.vstack([neg_scores, pos_scores])))
            
            if train_mode:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        for key in batch_metric:
            if train_mode:
                self.train_metric[key].append(np.mean(batch_metric[key]))
            else:
                if test_mode:
                    print("test:", key, np.mean(batch_metric[key]))
                else:
                    self.valid_metric[key].append(np.mean(batch_metric[key]))