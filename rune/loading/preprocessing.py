from glob import glob
import pandas as pd
import numpy as np
import re
import os
from itertools import chain

def split_tcrs(tcrab: str):
    tcrab = re.sub("TRA:", "", tcrab)
    tcrab = re.sub("TRB:", "", tcrab)
    return tcrab.split(';')

def split_train_test(data: pd.DataFrame, folder: str):
    test_ids = []
    valid_ids = []

    for i in data.iloc[:, :-3].columns:
        test_ids.append(np.random.choice(data[data[i] == 1].index, size=100, replace=False))

    for i, col in enumerate(data.iloc[:, :-3].columns):
        ids = np.setdiff1d(data[data[col] == 1].index, test_ids[i])
        valid_ids.append(np.random.choice(np.array(ids), size=100, replace=False))

    test_ids = list(chain.from_iterable(test_ids))
    valid_ids = list(chain.from_iterable(valid_ids))
    train_ids = np.setdiff1d(data.index, test_ids)
    train_ids = np.setdiff1d(train_ids, valid_ids)

    data.iloc[train_ids].to_csv(os.path.join(folder, "train.data"), index=False)
    data.iloc[valid_ids].to_csv(os.path.join(folder, "validation.data"), index=False)
    data.iloc[test_ids].to_csv(os.path.join(folder, "test.data"), index=False)

def preprocess(folder: str, save_path=None):
    data_paths = glob(folder + '/*matrix.csv')

    dflist = []
    for dpath in data_paths:
        df = pd.read_csv(dpath)
        dflist.append(df)

    df = pd.concat(dflist)
    df = df.drop_duplicates('cell_clono_cdr3_aa')
    ids = (df.cell_clono_cdr3_aa.apply(lambda x: x.count('TRA:')) == 1) & \
          (df.cell_clono_cdr3_aa.apply(lambda x: x.count('TRB:')) == 1)
    df = df[ids]
    df = df.reset_index(drop=True)

    cols = [i for i in df.select_dtypes('bool').sum().sort_values(ascending=False).index.to_series()[:4]]
    cols.append('cell_clono_cdr3_aa')

    data = df[cols]

    s = df.cell_clono_cdr3_aa.apply(split_tcrs)

    data = pd.concat([
        data,
        pd.DataFrame.from_items(zip(s.index, s.values), columns=['TRA', 'TRB'], orient='index')
    ], axis=1)

    if save_path:
        split_train_test(data, save_path)
    else:
        split_train_test(data, folder)
