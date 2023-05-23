import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

from .encoder import *

class RuneAutoencoder(nn.Module):
    """
    This networks encodes and decodes receptor sequences.
    """

    def __init__(self,
                 input_size,
                 hidden_size_gru=16,
                 n_layers=2,
                 dropout=0.0, 
                 **kwargs):
        super(RuneAutoencoder, self).__init__()

        self._input_size = input_size
        self._hidden_size_gru = hidden_size_gru
        self._n_layers = n_layers
        self._dropout = dropout
        
        encoder_params = {
            "input_size": self._input_size,
            "hidden_size": self._hidden_size_gru,
            "num_layers": self._n_layers,
            "batch_first": True,
            "bidirectional": True,
            "dropout": self._dropout
        }
        
        decoder_params = {
            "input_size": self._hidden_size_gru*2,
            "hidden_size": self._hidden_size_gru,
            "num_layers": self._n_layers,
            "batch_first": True,
            "bidirectional": True,
            "dropout": self._dropout
        }

        self.encoder = nn.LSTM(**encoder_params)
        self.encoder_head = nn.Sequential(nn.Linear(self._hidden_size_gru*2, self._hidden_size_gru*2), nn.LeakyReLU())
        self.decoder = nn.LSTM(**decoder_params)
        self.decoder_head = nn.Sequential(nn.Linear(self._hidden_size_gru*2, 21), nn.LeakyReLU())
    

    def forward(self, x, lengths):
        x = x.transpose(1, 2)

        # Encode the sequence
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.encoder(x)[1][0]
        x = torch.cat((x[-2, :, :], x[-1, :, :]), 1)
        x = self.encoder_head(x)
        
        # Decode the sequence
        x = self.decoder(x)[0]
        x = pad_packed_sequence(x, batch_first=True)
        x = self.decoder_head(x)
        x = F.softmax(x, 1)
        
        return x

    
    def _add_cnn(self, in_channels, out_channels):
        pass