
# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %%
data = pd.read_csv("D:/YZSeminerEgitimAgustos/Data/haftaHarcama.csv")
data


# %%
hafta = data[["Hafta"]]
harcama = data[["Harcama"]]


# %%
X_train, X_test, y_train, y_test = train_test_split(
    hafta, harcama, test_size=0.33)

X_train


# %%
y_train
# %%
# Model Kullan
loaded_model = joblib.load('gun_harcama_modeli.pkl')
sonuclar = loaded_model.score(X_test, y_test)
sonuclar

# %%

tahmin = loaded_model.predict(X_test)
tahmin

# %%
y_test

# %%
