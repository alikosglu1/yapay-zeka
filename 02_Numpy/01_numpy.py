
# %%
import numpy as np


# %%
a = np.random.randn(10)
print(a)


# %% [markdown]
# **Vektör**

# %%
Liste = [1, 2, 3]


# %%
b = np.array([1, 2, 3])
print(b)


# %%
print(b[1])

# %% [markdown]
# **Matrix**

# %%
c = np.array([[1, 2, 3], [44, 51, 67], [27, 56, 78]])
print(c)


# %%
c.shape


# %%
print(np.arange(0, 10))


# %%
np.arange(0, 10, 2)


# %%
np.zeros(3)


# %%
a = np.zeros((2, 3))
print(a)


# %%
d = np.ones(4)
print(d)


# %%
x = np.ones((4, 5))*8
print(x)


# %%
k = np.full((3, 3), 9)
print(k)


# %%
m = np.linspace(0, 10, 50)
print(m)


# %%
np.eye(4)


# %%
a = np.random.rand(3, 3)
print(a)


# %%
x = np.arange(25)
print(x)


# %%
z = x.reshape(5, 5)
print(z)

# %% [markdown]
# **Indeksleme**

# %%
x = np.arange(10)
print(x)


# %%
print(x[5])


# %%
print(x[2:7])


# %%
print(x[3:])


# %%
print(x[:5])


# %%
print(x)


# %%
x[:5] = 20
print(x)


# %%
x


# %%
x[:5] += 20
print(x)


# %%
mat = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

print(mat)


# %%
mat[0][2]


# %%
mat[1, 2]


# %%
mat[:2, 1:]

# %% [markdown]
# **Array Math**

# %%
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([[7, 8], [9, 10], [11, 12]])


# %%
print(x)


# %%
print(y)


# %%
print(x+y)


# %%
print(x-y)
print(np.subtract(x, y))


# %%
print(x*y)
# %%
x = np.array([[1, 2], [3, 4]])
y = np.array([[7, 8], [9, 10]])
x

# %%
y
# %%
np.dot(x, y)

# %%
print(x*y)

# %%


# %%
