# %%
from sklearn.impute import SimpleImputer
from IPython import get_ipython

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("D:/YZSeminerEgitimAgustos/Data/gelirVerileriEksik.csv")

print(data)

# %%
my_imputer = SimpleImputer()

Yas = data.iloc[:, 1:4]

Yas = data.iloc[:, 1:4].values
print(Yas)

# %%
# Sutün toplamı 1026 bölü 26 = 39,46 Eksik veriler ortalam ile yer değiştiriyor
data_with_imputed_values = pd.DataFrame(my_imputer.fit_transform(Yas))

print(data_with_imputed_values)

Yas[:, 0:4] = data_with_imputed_values.iloc[:, :].values

print(Yas[:, 0:4])

type(Yas)

# %%
yeni_data = data
yeni_data
print(type(yeni_data))
# Burada ise en çok tekrarlanan değer ile eksik veriler tamamlanıyor
imputer2 = SimpleImputer(strategy='most_frequent')
yeni_data = imputer2.fit_transform(yeni_data)


print(yeni_data)
type(yeni_data)

# %%
