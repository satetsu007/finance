# coding: utf-8

# libraryのインポートとか
import numpy as np
import pandas as pd
import sklearn.linear_model
import matplotlib.pyplot as plt
import os
import sys
from IPython.display import display
import seaborn as sns
import scipy
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LSTM
from keras.optimizers import Adam, SGD, RMSprop

plt.style.use("ggplot")

def main():
    print("set data.")
    n225 = pd.read_csv("./data/nikkei225_d.csv")
    usdjpy = pd.read_csv("./data/usdjpy_d.csv")

    n225.columns = ["Date", "n225_OPEN", "n225_HIGH", "n225_LOW", "n225_CLOSE"]
    usdjpy.columns = ["Date", "uj_OPEN", "uj_HIGH", "uj_LOW", "uj_CLOSE"]
    usdjpy.Date = pd.Series([i.replace("-", "/") for i in usdjpy.Date])

    df = pd.merge(usdjpy, n225)

    # df_norm = df.copy()
    # df_norm.iloc[:, 1:] = scipy.stats.zscore(df.iloc[:, 1:], axis=0)

    # df_diff = (df.iloc[:, 1:] - df.iloc[:, 1:].shift())[1:]
    # df_diff = pd.concat([df.Date[1:], df_diff], axis=1)
    
    # df_norm_diff = (df_norm.iloc[:, 1:] - df_norm.iloc[:, 1:].shift())[1:]
    # df_norm_diff = pd.concat([df.Date[1:], df_norm_diff], axis=1)

    target_label = "n225_CLOSE"
    length_for_times = 5
    after_times = 5
    batch_size = 128
    epochs = 10
    

    X, y = set_data(df, target_label,
                    length_for_times=length_for_times, after_times=after_times)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    n_in = X_train[1]
    n_out = 1

    print("define model.")
    model = set_model(n_in, n_out)

    print("train model.")
    model.fit(X_train, y_train, batch_size=batch_size, validation_split=0.2, epochs=epochs)
    
    print("test model.")
    y_pred = model.predict(X_test, batch_size=32)

    print("plot data.")
    fig, ax1 = plt.subplots()
    ax1.plot(y_test, color="red")
    ax2 = ax1.twinx()  # 2つのプロットを関連付ける
    ax2.plot(y_pred, color="blue")
    plt.show()


def set_data(df, target_label, length_for_times=1, after_times=1):
    """
    input: DataFrame, target_label, length_for_times, after_times
    output: X, y
    """

    X = []
    y = []

    for i in range(len(df)-length_for_times-after_times):
        X.append(df.iloc[i:i+length_for_times, 1:].as_matrix().flatten())

    for i in range(len(df)-length_for_times-after_times):
        y.append(df[target_label].iloc[i-1+length_for_times+after_times])
   
    return np.array(X), np.array(y)

def set_model(n_in, n_out):
    """
    input: n_in, n_our
    output: model
    """

    model = Sequential()
    model.add(Dense(128, input_shape=(n_in, )))
    model.add(Activation("sigmoid"))
    model.add(Dense(128))
    model.add(Activation("sigmoid"))
    model.add(Dense(n_out))
    model.add(Activation("sigmoid"))
    
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    return model

if __name__=="__main__":
    main()