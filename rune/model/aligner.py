import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
    

class RuneAligner(nn.Module):
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
                 hidden_size_gru=16,
                 n_layers=2,
                 dropout=0.0, 
                 **kwargs):
        super(RuneAligner, self).__init__()

        self._input_size = input_size
        self._hidden_size_gru = hidden_size_gru
        self._n_layers = n_layers
        self._dropout = dropout
        
        gru_params = {
            "input_size": self._input_size,
            "hidden_size": self._hidden_size_gru,
            "num_layers": self._n_layers,
            "batch_first": True,
            "bidirectional": True,
            "dropout": self._dropout
        }

        self.gru = nn.LSTM(**gru_params)
        self.similariry_net = nn.Sequential(nn.Linear(self._hidden_size_gru*2, 1), nn.LeakyReLU())
        self.penalty_gap = nn.Parameter(torch.tensor([-1.0]).reshape(-1,1))
    

    def forward(self, batch_neg, len_neg, batch_pos, len_pos):
        def _get_embeddings(x, lengths):
            x = x.transpose(1, 2)
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            x = self.gru(x)[0]
            x = pad_packed_sequence(x, batch_first=True)[0]
            x = torch.cat((x[-2, :, :], x[-1, :, :]), 1)
            return x
    
        x_neg = _get_embeddings(batch_neg, len_neg)
        x_pos = _get_embeddings(batch_pos, len_pos)
        
        scores_prev = [torch.tensor([self.penalty_gap * i] * len(batch_neg)).reshape(-1,1) for i in range(len_neg.max().item()+1)]
        scores_cur = [torch.tensor([0.0] * len(batch_neg)).reshape(-1,1)]
            
        for neg_pos in range(1, len_neg.max()[0]+1):
            for pos_pos in range(1, len_pos.max()[0]):
#                 merged_embed = torch.cat((x_neg[:, neg_pos, :], x_pos[:, pos_pos, :]), 1)
                merged_embed = (x_neg[:, neg_pos, :], x_pos[:, pos_pos, :]).mul(.5)
                print(merged_embed.shape)
                out = torch.max(torch.max(scores_prev[neg_pos-1] + self.similariry_net(merged_embed), 
                                          scores_prev[neg_pos] + self.penalty_gap), 
                                scores_cur[pos_pos-1] + self.penalty_gap)
                scores_cur.append(out)
            scores_prev = scores_cur
            scores_cur = [torch.tensor([self.penalty_gap * fc_i] * len(x)).reshape(-1,1)]
                
        # sum of scores = mult of log-probs
        # p = x*y => exp(logx + logy)
        return scores_prev[-1] / lengths.view(-1,1)
        return torch.sigmoid(torch.exp(scores_prev[-1] / lengths.view(-1,1)))