# coding: utf-8

import numpy as np
import pandas as pd
import sklearn.linear_model
import matplotlib.pyplot as plt
import os
import sys
from IPython.display import display
import seaborn as sns
import scipy
plt.style.use("ggplot")

def main():
    n225 = pd.read_csv("./data/nikkei225_d.csv")
    usdjpy = pd.read_csv("./data/usdjpy_d.csv")

    n225.columns = ["Date", "n225_OPEN", "n225_HIGH", "n225_LOW", "n225_CLOSE"]
    usdjpy.columns = ["Date", "uj_OPEN", "uj_HIGH", "uj_LOW", "uj_CLOSE"]
    usdjpy.Date = pd.Series([i.replace("-", "/") for i in usdjpy.Date])

    df = pd.merge(usdjpy, n225)

    df_norm = df.copy()
    df_norm.iloc[:, 1:] = scipy.stats.zscore(df.iloc[:, 1:], axis=0)

    df_diff = (df.iloc[:, 1:] - df.iloc[:, 1:].shift())[1:]
    df_diff = pd.concat([df.Date[1:], df_diff], axis=1)
    
    df_norm_diff = (df_norm.iloc[:, 1:] - df_norm.iloc[:, 1:].shift())[1:]
    df_norm_diff = pd.concat([df.Date[1:], df_norm_diff], axis=1)

def set_data(df, target_label, length_for_times, after_times):
    """
    input: DataFrame, target_label, length_for_times, after_times
    output: X, y
    """

    df_tmp = df.shift()
    

if __name__=="__main__":
    main()