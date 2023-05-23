import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

from .encoder import *
    

class RuneClassifier(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 fc_layers=1,
                 n_filters=32,
                 kernel_size=3,
                 cnn=2,
                 hidden_size_gru=16,
                 n_layers=2,
                 **kwargs):
        super(RuneClassifier, self).__init__()
        self.fc_n = fc_layers - 1
        self.fc_in = 2 * hidden_size_gru
        self.fc_net = self._make_fc_net(output_size)
        
#         self.model_list = nn.ModuleList()
#         for i in range(2):
#             encoder = RuneEncoder(input_size,
#                                      n_filters,
#                                      kernel_size,
#                                      cnn,
#                                      hidden_size_gru,
#                                      n_layers,
#                                      dropout)
#             fc_net = self._make_fc_net()
#             self.model_list.append(nn.ModuleList([encoder, fc_net]))
        self.encoder = RuneEncoder(input_size,
                                     n_filters,
                                     kernel_size,
                                     cnn,
                                     hidden_size_gru,
                                     n_layers,
                                     **kwargs)
        
    def _make_fc_net(self, output_size):
        fc_layers = []
        
        for i in range(self.fc_n-1):
            fc_layers.append(nn.Linear(self.fc_in, self.fc_in))
            torch.init.kaiming_uniform_(fc_layers[-1].weight)
            fc_layers.append(nn.LeakyReLU())
            
        if output_size == 2:
            fc_layers.append(nn.Linear(self.fc_in, 1))
            torch.nn.init.kaiming_uniform_(fc_layers[-1].weight)
            fc_layers.append(nn.Sigmoid())
        else:
            fc_layers.append(nn.Linear(self.fc_in, output_size))
            torch.nn.init.kaiming_uniform_(fc_layers[-1].weight)

        
        fc_net = nn.Sequential(*fc_layers)
        
        return fc_net
        
        
    def forward(self, batch, lens):
        x = self.encoder(batch, lens)
        x = self.fc_net(x)
        return x
#         res = []
#         for i in range(len(self.model_list)):
#             x = self.model_list[i][0](batch, lens)
#             x = self.model_list[i][1](x)
#             res.append(x)
#         res = torch.cat(res, 1).mean(1)
        
#         return res