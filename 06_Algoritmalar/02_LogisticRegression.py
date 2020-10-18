
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %%
# plot sigmoid function
a = np.linspace(-10, 10, 100)
c = 0.5*a  # Buradaki çarpanı 0.5-5 arasında değiştir ve sonucu gör
b = 1.0 / (1.0 + np.exp(-c))

plt.plot(a, b, 'r-', label='logit, sigmoid')
plt.legend(loc='lower right')


# %% [markdown]
# # Yaşa Göre Evinin olup olmadığını bulacağız


# %%
data = pd.read_csv("D:/YZSeminerEgitimAgustos/Data/gelirVerileri3.csv")
data


# %%
X = data.iloc[:, 0:1].values
Y = data.iloc[:, 3:4].values


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=0)


# %%
LogReg = LogisticRegression(random_state=0)
LogReg.fit(X_train, y_train)


# %%
y_pred = LogReg.predict(X_test)
y_pred


# %%
y_test


# %%
(X_test)


# %%
X_test_sorted = np.sort(X_test, axis=0)
X_test_sorted


# %%
#  predict_proba iki sütunlu değer döndürüyor ilk sütun olmama olasılığı, ikinci sutün olma olasılığı
m = LogReg.predict_proba(X_test_sorted)
m[:, 1]


# %%
# plotting fitted line
plt.scatter(data.yas, Y,  color='black')
plt.yticks([0.0, 0.5, 1.0])
plt.plot(X_test_sorted, m[:, 1], color='blue', linewidth=3,  marker='o',
         markeredgecolor='red', markerfacecolor="red", markersize=8)

plt.ylabel('Ev')
plt.xlabel('Yaş')
plt.show()
