import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

class RuneEncoder(nn.Module):
    """
    This networks encodes receptor sequences.
    
    Args:
        input_size:
        n_filters:
        kernel_size:
        cnn:
        hidden_size_gru:
        n_layers:

    Examples:

    """

    def __init__(self,
                 input_size,
                 n_filters=32,
                 kernel_size=3,
                 cnn=2,
                 hidden_size_gru=16,
                 n_layers=2,
                 dropout_rnn=0.0, 
                 dropout_cnn=0.0,
                 **kwargs):
        super(RuneEncoder, self).__init__()

        self._input_size = input_size
        self._n_filters = n_filters
        self._kernel_size = kernel_size
        self._cnn = cnn
        self._hidden_size_gru = hidden_size_gru
        self._n_layers = n_layers

        conv_params = {
            "in_channels": self._input_size,
            "out_channels": self._n_filters,
            "kernel_size": self._kernel_size,
            "padding": self._kernel_size // 2
        }

        if self._cnn < 1:
            self._n_filters = self._input_size
        else:
            conv_params["kernel_size"] = 1
            
            conv_layers = [nn.Conv1d(**conv_params)]
            conv_layers.append(nn.LeakyReLU())
#             conv_layers.append(nn.GELU())
            if dropout_cnn:
                conv_layers.append(nn.Dropout(dropout_cnn))                
            self._cnn_start = nn.Sequential(*conv_layers)
            torch.nn.init.kaiming_uniform_(self._cnn_start[0].weight)
            
            conv_layers = []
            if cnn > 1:
                conv_params["kernel_size"] = 5
                conv_params["in_channels"] = self._n_filters
                
                for i in range(self._cnn - 1):
                    conv_layers.append(nn.Conv1d(**conv_params))
                    torch.nn.init.kaiming_uniform_(conv_layers[-1].weight)
                    
                    conv_layers.append(nn.LeakyReLU())
                    if dropout_cnn:
                        conv_layers.append(nn.Dropout(dropout_cnn))

            self.conv_net = nn.Sequential(*conv_layers)
        
        gru_params = {
            "input_size": self._n_filters,
            "hidden_size": self._hidden_size_gru,
            "num_layers": self._n_layers,
            "batch_first": True,
            "bidirectional": True,
            "dropout": dropout_rnn
        }

        self.gru = nn.LSTM(**gru_params)
    
        self.attn = nn.Sequential(nn.Linear(gru_params["hidden_size"]*2, gru_params["hidden_size"]*2),
                                  nn.LeakyReLU(),
                                  nn.Linear(gru_params["hidden_size"]*2, 1),
                                  nn.Sigmoid())
#         self.final = nn.Sequential(nn.Linear(gru_params["hidden_size"]*2, gru_params["hidden_size"]*2), 
#                                    nn.LeakyReLU())
    

    def forward(self, x, lengths):        
        ####################
        # CNN from the start
        if self._cnn:
#             out = self._cnn_start(x)
#             x = self.conv_net(out)
#             x = F.leaky_relu(x + out)
            x = self._cnn_start(x)
            x = self.conv_net(x)
        ####################
        
        # LSTM
        x = x.transpose(1, 2)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = self.gru(x)[1][0]
        x = torch.cat((x[-2, :, :], x[-1, :, :]), 1)
        
        # Post-LSTM FC layers
#         x = self.final(x)
        ####################

        # LSTM + Attention
#         x = x.transpose(1, 2)
#         x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
#         x = self.gru(x)[0]
#         x, lens_vec = pad_packed_sequence(x, batch_first=True)
#         maxlen = lens_vec.max()
#         mask = torch.arange(lens_vec.max())[None, :] < lens_vec[:, None]
#         if x.get_device() == -1:
#             device = "cpu"
#         else:
#             device = "cuda"
#         mask = mask.to(device)
#         attn_weights = self.attn(x).masked_fill(mask.unsqueeze(2) == False, 0)
#         attn_weights = F.softmax(attn_weights, 1) # SOFTMAX #
#         x = x * attn_weights
#         x = x.sum(1)
        ####################
        
        return x

    
    def _add_cnn(self, in_channels, out_channels):
        pass