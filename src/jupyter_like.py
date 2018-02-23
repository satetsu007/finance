# coding: utf-8

#%%
# ライブラリのインポート
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

#%%
# データのロードと軽い加工
n225 = pd.read_csv("./data/nikkei225_d.csv")
usdjpy = pd.read_csv("./data/usdjpy_d.csv")

n225.columns = ["Date", "n225_OPEN", "n225_HIGH", "n225_LOW", "n225_CLOSE"]
usdjpy.columns = ["Date", "uj_OPEN", "uj_HIGH", "uj_LOW", "uj_CLOSE"]
usdjpy.Date = pd.Series([i.replace("-", "/") for i in usdjpy.Date])

df = pd.merge(usdjpy, n225)

df_norm = df.copy()
df_norm.iloc[:, 1:] = scipy.stats.zscore(df.iloc[:, 1:], axis=0)

display(df.head())

#%%
# データの概形
df.plot(x="Date", y="uj_CLOSE", figsize=[10, 4])
df.plot(x="Date", y="n225_CLOSE", figsize=[10, 4])

#%%
# 変動のプロット
df_diff = (df.iloc[:, 1:] - df.iloc[:, 1:].shift())[1:]
df_diff = pd.concat([df.Date[1:], df_diff], axis=1)
df_diff.head()
# df_diff.plot(x="Date", y="uj_CLOSE", figsize=[10, 4])
# df_diff.plot(x="Date", y="n225_CLOSE", figsize=[10, 4])

#%%
# 変動のプロット(標準化)
df_norm_diff = (df_norm.iloc[:, 1:] - df_norm.iloc[:, 1:].shift())[1:]
df_norm_diff = pd.concat([df.Date[1:], df_norm_diff], axis=1)
df_norm_diff.head()
# df_norm_diff.plot(x="Date", y="uj_CLOSE", figsize=[10, 4])
# df_norm_diff.plot(x="Date", y="n225_CLOSE", figsize=[10, 4])

#%%
# 各特徴量の関係をプロット(重い)
# sns.pairplot(df.iloc[:, 1:])
# sns.pairplot(df_norm.iloc[:, 1:])
# sns.pairplot(df_diff.iloc[:, 1:])
# sns.pairplot(df_norm_diff.iloc[:, 1:])

#%%
# 各データフレームの統計量を表示
display(df.describe())
display(df_norm.describe())
display(df_diff.describe())
display(df_norm_diff.describe())

#%%
X = []
X_tmp = []
y = []
y_tmp = []

length_for_times = 5
after_times = 5
target_label = "n225_CLOSE"

for i in range(len(df)-length_for_times-after_times):
    X.append(df.iloc[i:i+length_for_times, 1:].as_matrix().flatten())
    # X_tmp.append(df.iloc[i:i+length_for_times, 1:])
X = np.array(X)

# X_tmp[:5]

for i in range(len(df)-length_for_times-after_times):
    y.append(df[target_label].iloc[i-1+length_for_times+after_times])
    # y_tmp.append(df[target_label].iloc[i-1+length_for_times+after_times])
    # X_tmp.append(df.iloc[i-1+length_for_times+after_times])

y =  np.array(y)

print(X.shape)
print(y.shape)



display(df.iloc[:length_for_times + after_times])
#%%