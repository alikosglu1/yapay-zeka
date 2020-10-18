
# %%
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC


# %%


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

sc = StandardScaler()


# %%
X_train_scaler = sc.fit_transform(X_train)
X_test_scaler = sc.fit_transform(X_test)


# %%
svc = SVC(kernel='linear')
svc.fit(X_train_scaler, y_train)


# %%
y_pred3 = svc.predict(X_test_scaler)
y_pred3
# %%
y_test

# %%
ConMat3 = confusion_matrix(y_test, y_pred3)
ConMat3


# %%
