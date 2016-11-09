# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import drill_utilities as du
import warnings
warnings.filterwarnings('ignore')

# Load set of waves
# Load Matrice and target
X = np.load("icu_matrix5.pyriemann.npy")
y_true = np.load("icu_target5.pyriemann.npy")
ids = np.load("icu_ids5.pyriemann.npy")
print y_true
print(len(y_true[y_true==True]))
X.shape, np.size(y_true)

# Mortality
print(len(y_true[y_true==True]))
print(float(len(y_true[y_true==True])) / float(len(y_true))* 100.)

# Min-Max values normalization
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler

original_X = X
for case in range(X.shape[0]):
    try:
        X[case] = scale(X[case])
    except:
        print(case, ids[case])
        print(X[case])
