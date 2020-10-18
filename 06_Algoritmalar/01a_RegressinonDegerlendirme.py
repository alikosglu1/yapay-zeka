# %%
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# %%

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
    hafta, harcama, test_size=0.33, random_state=0)


# %%
X_train


# %%
y_train


# %%
lrModel = LinearRegression()
lrModel.fit(X_train, y_train)


# %%
tahmin = lrModel.predict(X_test)
tahmin


# %%
y_test


# %%

plt.scatter(X_train, y_train)
plt.plot(X_test, tahmin, '-o', color="red")


# %%

print("R2 score : %.5f" % r2_score(y_test, tahmin))


# %%
print("Mean squared error: %.5f" % mean_squared_error(y_test, tahmin))

# %%
lrModel.score(X_test, y_test)


# %%
