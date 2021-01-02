import os
import glob
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

def get_data(PATH,IMG_SIZE=256):
    X = []
    for img in glob.glob(PATH+"/*"):
        x = np.array(Image.open(img))
        x = cv2.resize(x, (IMG_SIZE, IMG_SIZE))
        X.append(x)
    return np.array(X)


def getNearestValue(z_database, z):
    """ 
    概要: リストからある値に最も近いインデックスを返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """
    assert z.shape[0]==z_database.shape[1], "z dim error"
    
    # Calculate z score distance
    # squared distance
    n_score = np.abs((z_database - z)**2).sum(axis=1)
    sort_idx = np.argsort(n_score)
    
    return sort_idx


from scipy.spatial import distance
def mahalanobis_equation(Z,z):

    # 分散共分散行列を計算
    cov = np.cov(Z.T)
    # 分散共分散行列の逆行列を計算
    cov_i = np.linalg.pinv(cov)

    D_list = []
    for i in range(len(Z)):
        # 2つの標本のマハラノビス距離を計算する
        d = distance.mahalanobis(Z[i], z, cov_i)
        D_list.append(d)
        # print("マハラノビス距離の計算結果: %1.2f" % d)
    D_list = np.array(D_list)
    
    return np.argsort(D_list)