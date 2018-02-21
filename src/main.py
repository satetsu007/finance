# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import keras

def main():
    n225 = pd.read_csv("./data/nikkei225_d.csv")
    usdjpy = pd.read_csv("./data/usdjpy_d.csv")
    
    