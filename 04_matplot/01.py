

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
x = [1.3, 2.9, 3.1, 4.7, 5.6, 6.5, 7.4, 8.8, 9.2, 10]
y = [95, 42, 69, 11, 49, 32, 74, 62, 25, 32]


plt.plot(x, y)
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")


# %%
y = [95, 42, 69, 11, 49, 32, 74, 62, 25, 32]


plt.plot(y, 'g')


# %%


plt.plot(y, 'ro-')


# %%

plt.plot(y, 'b*--')


# %%

plt.plot(y, 'g-')
plt.plot(y, 'ro')


# %%
y = [95, 42, 69, 11, 49, 32, 74, 62, 25, 32]
y2 = [35, 52, 96, 77, 36, 66, 50, 12, 35, 63]


plt.plot(y, 'go-')
plt.plot(y2, 'b*--')


# %%

plt.plot(y, 'go-', y2, 'b*--')


# %%
arr = np.random.normal(size=30)
plt.plot(arr, color="teal", marker="*", linestyle="dashed")


# %%

data = pd.read_csv("https://bit.ly/2WcsJE7", index_col=0, parse_dates=True)
data.head()


# %%
data.Volume.iloc[:100].plot()


# %%
data[['AdjOpen', "AdjClose"]][:50].plot()


# %%
x = [1.3, 2.9, 3.1, 4.7, 5.6, 6.5, 7.4, 8.8, 9.2, 10]
y = [95, 42, 69, 11, 49, 32, 74, 62, 25, 32]
color = np.random.rand(10)

plt.scatter(x, y, c=color)


# %%
plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 10], "go", label="Yeşil Noktalar")
plt.plot([1, 2, 3, 4, 5], [2, 3, 4, 5, 11], "b*", label="Mavi Yıldızlar")
plt.title("Basit ScatterPlot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# %%
labels = ["Frogs", "Cats", "Dogs", "Mouse"]
size = [15, 30, 45, 10]

fig1, ax1 = plt.subplots()
explode = (0, 0.1, 0, 0.1)
ax1.pie(size, explode=explode, labels=labels)
plt.show()


# %%
