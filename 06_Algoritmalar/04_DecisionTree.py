
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from IPython import get_ipython
from sklearn import tree


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


# %%
data = pd.read_csv("D:/YZSeminerEgitimAgustos/Data/gelirVerileri3.csv")
data


# %%
X = data.iloc[:, 0:3].values
Y = data.iloc[:, 3:4].values


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=0)


# %%
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)


# %%
X_test


# %%
y_test


# %%
y_pred3 = dtc.predict(X_test)
y_pred3


# %%
ConMat3 = confusion_matrix(y_test, y_pred3)
ConMat3


# %%
# plt.scatter(X_train[:,0:1], X_train[:,1:2], c=y_train,  cmap="rainbow")
plt.scatter(X_test[:, 0:1], X_test[:, 1:2], c=y_test,  cmap="rainbow")
plt.xlabel("yas")
plt.ylabel("gelir")
plt.show


# %%


fig, ax = plt.subplots(figsize=(30, 20))
tree.plot_tree(dtc,  fontsize=30,  feature_names=[
               "yas", "Gelir", "Tecrube"])


# %%
dtc.predict([[34, 15, 19]])


# %%
print(X_train)


# %%
print(y_train)


# %%
