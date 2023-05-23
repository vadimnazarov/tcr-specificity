import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset


class SeqProcessor:
    """
    A class that processes amino acid sequences for Machine Learning
    
    Attributes:
        sequence: a list of amino acid sequences
        encoding: encoding set for amino acids, includes
                - Kidera factors
                - Simple index
                - One-hot encoding
        min_len: filter sequences of length less than min_len
        max_len: filter sequences of length greater than max_lin
    """
    
    def __init__(self, sequences, labels, encoding="onehot", min_len=9, max_len=19):
        self.min_len = min_len
        self.max_len = max_len
        self.encoding_dict = self._load_dict(encoding)
        self.sequences, self.labels = self._filter_length(sequences, labels)


    def _load_dict(self, encoding):
        filename = "one_hot.pkl"
        
        if encoding in ["onehot", "one_hot"]:
            filename = "one_hot.pkl"
        elif encoding == "kidera":
            filename = "kidera.pkl"
        elif encoding == "atchley":
            filename = "atchley.pkl"
        elif encoding == "index":
            filename = "index.pkl"
        else:
            print("Unknown key:", encoding, " Returning the one-hot dictionary.")

        with open("features/" + filename, "rb") as file:
            return pickle.load(file)
        
        
    def _filter_length(self, sequences, labels):
        def _get_len(seq_vec):
            return np.array([len(x) for x in seq_vec])
        
        lens_vec = _get_len(sequences)
        logic = (lens_vec >= self.min_len) & (lens_vec <= self.max_len)
        return sequences[logic], labels[logic]
        
        
    def _seq2vec(self, seq_list, max_len=None):       
        # N, C, L
        # WIP: Do we need this check?
        # Action required
        if max_len <= 0:
            max_len = max([len(seq) for seq in seq_list])
            
        res = torch.zeros((len(seq_list), len(self.encoding_dict["A"]), max_len))
        
        for seq_i, seq in enumerate(seq_list):
            for aa_i, aa in enumerate(seq.upper()):
                try:
                    res[seq_i, :, aa_i] = self.encoding_dict[aa]
                except KeyError as e:
                    raise Exception(f"{e} is not a valid amino acid")
                
        # Squeeze if last dimension is only one?
        # Good for the index encoding for embeddings
        
        return res
        

    def get_neg_pos(self, max_n=0, seed=None):
        neg = self.sequences[self.labels == False]
        pos = self.sequences[self.labels == True]
        
        neg = pd.Series(neg).str.slice(start = 2, stop = -2).reset_index(drop=True)
        pos = pd.Series(pos).str.slice(start = 2, stop = -2).reset_index(drop=True)
        
#         for i in range(len(neg)):
#             position = np.random.randint(len(neg[i]) - 4)
#             neg[i] = neg[i][:position] + "AM" + neg[i][position:position+1] + "K" + neg[i][position+2:]
#         for i in range(len(pos)):
#             position = np.random.randint(len(pos[i]) - 4)
#             pos[i] = pos[i][:position] + "L" + pos[i][position:position+1] + "HQ" + pos[i][position+2:]
            
        np.random.seed(seed)
        if max_n:
            neg = neg[np.random.choice(len(neg), max_n)].reset_index(drop=True)
            pos = pos[np.random.choice(len(pos), max_n)].reset_index(drop=True)
        else:
            np.random.shuffle(neg.values)
            np.random.shuffle(pos.values)
        
        max_len = self.max_len - 2
        
        label_neg = torch.tensor([0] * len(neg), dtype=torch.float)
        len_neg = torch.from_numpy(neg.str.len().values).float()
        vec_neg = self._seq2vec(neg, max_len)
        ds_neg = TensorDataset(vec_neg, len_neg.view(-1, 1), label_neg.view(-1, 1))

        label_pos = torch.tensor([1] * len(pos), dtype=torch.float)
        len_pos = torch.from_numpy(pos.str.len().values).float()
        vec_pos = self._seq2vec(pos, max_len)
        ds_pos = TensorDataset(vec_pos, len_pos.view(-1, 1), label_pos.view(-1, 1))

        return ds_neg, ds_pos
