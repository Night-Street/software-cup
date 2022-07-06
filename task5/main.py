from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import k_means, kmeans_plusplus, dbscan
from sklearn.preprocessing import LabelEncoder

from utils import calc_rfm, process_csv, setup_seed


def check_n_clusters(n_clusters: int):
    assert n_clusters > 0, "n_clusters should be greater than 0."
    if n_clusters > 7:
        print("n_clusters should NOT be greater than 7, which is to be clipped down to 7.")
    return min(7, n_clusters)


def main(n_clusters: int = 4):
    n_clusters = check_n_clusters(n_clusters)

    setup_seed()
    rfm_df = process_csv()
    for column_name in rfm_df.columns:
        _m, _s = np.mean(rfm_df[column_name]), np.std(rfm_df[column_name])
        rfm_df[column_name] = (rfm_df[column_name] - _m) / _s

    rfm_npa = rfm_df.to_numpy()
    _, labels, _ = k_means(rfm_npa, n_clusters)
    d = dict()
    for i in range(len(labels)):
        try:
            d[labels[i]].append(i)
        except:
            d[labels[i]] = [i]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for l, c, m in zip(range(n_clusters),
                       ["orange", "purple", "#1E90FF", "#7FFF00", "#FF69B4", "#FF6347", "#8B8682"],
                       ["o", "^", "*", "+", ".", "v", "s"]):
        ax.scatter(xs=rfm_npa[d[l], 0], ys=rfm_npa[d[l], 1], zs=rfm_npa[d[l], 2], c=c, marker=m, s=40)
        # ax.scatter(xs=centroids[l, 0], ys=centroids[l, 1], zs=centroids[l, 2], c="black", marker=m, s=80)
    ax.set(xlabel="R", ylabel="F", zlabel="M")
    plt.legend(range(1, 1 + n_clusters))
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--k", default="4")
    args = parser.parse_args()
    k = int(args.k)
    main(k)
