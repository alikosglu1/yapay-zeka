# %%
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
data = pd.read_csv("D:/YZSeminerEgitimAgustos/Data/gelirVerileriEksik.csv")

data
# %%
sehir = data.iloc[:, 0:1].values
sehir
# %%
lbe = LabelEncoder()

sehir[:, 0] = lbe.fit_transform(sehir[:, 0])
sehir[:, 0]
# %%
# ## **OneHot encoder**
sehir = data.iloc[:, 0:1].values
sehir

# %%

# Create one-hot encoder
one_hot = LabelBinarizer()

# One-hot encode feature
one_hot.fit_transform(sehir)
sehir = one_hot.fit_transform(sehir)
sehir
# %%
# View feature classes
print(one_hot.classes_)
# %%
# Reverse one-hot encoding
sehir = one_hot.inverse_transform(sehir)

sehir

# %%
