
# %%
import pandas as pd

veriler = pd.read_table("http://bit.ly/chiporders")


# %%
veriler.head()


# %%
izleyiciler = pd.read_table("http://bit.ly/movieusers")
izleyiciler


# %%
user_cols = ["user_id", "age", "gender", "accupation", "zip_code"]
izleyiciler = pd.read_table(
    "http://bit.ly/movieusers", names=user_cols, sep='|')
izleyiciler


# %%
ufo = pd.read_csv("http://bit.ly/uforeports")
ufo


# %%
a = "City"

ufo[a]


# %%
ufo.City


# %%
ufo["Colors Reported"]


# %%
ufo.shape


# %%
ufo.describe()


# %%
type(ufo)


# %%
ufo.columns


# %%
ufo.columns = ufo.columns.str.replace(' ', '_')


# %%
ufo.columns

# %% [markdown]
# ## Drop

# %%
ufo3 = ufo.drop(["City", "State"], axis=1, inplace=True)


# %%
ufo


# %%
ufo = pd.read_csv("http://bit.ly/uforeports")
ufo


# %% [markdown]
# ## Sorting

# %%
movies = pd.read_csv("http://bit.ly/imdbratings")
movies.head()


# %%
movies["title"].sort_values()


# %%
movies["title"].sort_values(ascending=False)


# %%
movies.sort_values("title", ascending=False)


# FİLTRELEME

# %%
# 2. Satır bütün sütunlar
movies[2:3]


# %%
# 1-4 arası satırlar, 3 e kadar olan sütun
movies.iloc[1:4, 1:3]


# %%
movies[movies.duration >= 200]


# %%
movies[(movies.duration >= 200) & (movies.genre == "Drama")]


# %%
movies[(movies.duration >= 200) | (movies.genre == "Drama")]


# %%
