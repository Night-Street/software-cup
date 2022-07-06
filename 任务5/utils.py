import datetime
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


def setup_seed(seed=0):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    from torch.backends import cudnn
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def calc_rfm(df: pd.DataFrame):
    _npa = df.to_numpy()

    now = datetime.datetime.now().timestamp()
    r = now - datetime.datetime.strptime(_npa[-1, 0], "%Y-%m-%d").timestamp()

    f = len(df)

    m = np.sum(_npa[:, -1])

    return r, f, m


def process_csv():
    # 计算R,F,M的值
    df = pd.read_csv("../dataset/cph.csv", index_col=0)
    le = LabelEncoder()
    df.index = le.fit_transform(df.index)
    df.index.name = "用户编号"
    # print(df.info())
    # print(df.index.name) 用户编号
    grouped_df = df.groupby("用户编号")
    rfm_df_data = []
    for _id, _sdf in grouped_df:
        r, f, m = calc_rfm(_sdf)
        s = pd.Series(np.zeros(3, ), index=["R", "F", "M"], dtype=float, name=_id)
        s["R"] = r
        s["F"] = f
        s["M"] = m
        rfm_df_data.append(s)
    rfm_df = pd.DataFrame(rfm_df_data)
    return rfm_df
