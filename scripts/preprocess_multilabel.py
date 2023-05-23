import re
import pandas as pd


def split_tcrs(tcrab: str):
    tcrab = re.sub("TRA:", "", tcrab)
    tcrab = re.sub("TRB:", "", tcrab)
    return tcrab.split(';')

def merge_matrices(donor: str):

    fmpath = "../data/raw/full_frequency_matrix_{}.csv.gz".format(donor)
    
    df = pd.read_csv(fmpath, compression="gzip", index_col=0)
    df = df.loc[df.sum(axis=1).loc[lambda x: (x >= 3) & (x <= 500)].index]
    df = df.applymap(lambda x: 0 if x==0 else 1)

    bmpath = "../data/raw/vdj_v1_hs_aggregated_{}_binarized_matrix.csv.gz".format(donor)

    bm = pd.read_csv(bmpath, compression="gzip")
    bm = bm.drop(bm.select_dtypes("bool").columns, axis=1)
    bm = bm.drop(bm.select_dtypes("float").columns, axis=1)
    bm = bm.drop('cell_clono_cdr3_nt', axis=1)

    ids = (bm.cell_clono_cdr3_aa.apply(lambda x: x.count('TRA:')) == 1) & \
        (bm.cell_clono_cdr3_aa.apply(lambda x: x.count('TRB:')) == 1)
    bm = bm[ids]
    bm = bm.reset_index(drop=True)

    s = bm.cell_clono_cdr3_aa.apply(split_tcrs)

    bm = pd.concat([
        bm,
        pd.DataFrame.from_items(zip(s.index, s.values), columns=['TRA', 'TRB'], orient='index')
    ], axis=1)

    bm = bm.set_index("barcode")
    res = pd.concat([bm, df], axis=1, join="inner")
    res = res.reset_index()

    return res

def main():
    dflist = []
    for i in range(1,5):
        dflist.append(merge_matrices('donor%i' % i))

    res = pd.concat(dflist)
    res = res.drop_duplicates("cell_clono_cdr3_aa")
    res.to_csv("../data/train_multilabel.data", index=False)

if __name__ == "__main__":
    main()