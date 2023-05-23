import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
    
    
class RuneRSP(nn.Module):
    
    def __init__(self,
                 input_size,
                 fc_layers=2,
                 n_filters=32,
                 kernel_size=3,
                 cnn=2,
                 hidden_size_gru=16,
                 n_layers=2,
                 dropout=0.0, 
                 **kwargs):
        super(RuneRSP, self).__init__()
        self.fc_n = fc_layers
        self.fc_in = 4 * hidden_size_gru
        self.fc_net = self._make_fc_net()
        self.encoder = RuneEncoder(input_size,
                                     n_filters,
                                     kernel_size,
                                     cnn,
                                     hidden_size_gru,
                                     n_layers,
                                     dropout)
        
    def _make_fc_net(self):
        fc_layers = []
        
        for i in range(self.fc_n-1):
            fc_layers.append(nn.Linear(self.fc_in, self.fc_in))
            fc_layers.append(nn.LeakyReLU())
            
        fc_layers.append(nn.Linear(self.fc_in, 1))
        fc_layers.append(nn.Sigmoid())
        
        fc_net = nn.Sequential(*fc_layers)
        
        return fc_net
    
    
    def _make_embeddings(self, vec_neg, len_neg, vec_pos, len_pos):
        out_neg = self.encoder(vec_neg, len_neg)
        out_pos = self.encoder(vec_pos, len_pos)
        
        out_vec = torch.cat((out_neg, out_pos), 1)
        
        return out_vec
        
        
    def forward(self, vec_neg, len_neg, vec_pos, len_pos):
        emb = self._make_embeddings(vec_neg, len_neg, vec_pos, len_pos)
        res = self.fc_net(emb)
        
        return res