import pandas as pd

df = pd.read_csv("panda/pokemon_data.csv")

## Add a new column
df["Total"] = (
    df["HP"]
    + df["Attack"]
    + df["Defense"]
    + df["Sp. Atk"]
    + df["Sp. Def"]
    + df["Speed"]
)
print(df.head(5))

## Add a new column way 2
# Axis = 0, vertical
# Axis = 1, horizontal
# df["Total"] = df.iloc[:, 4:10].sum(axis=1)
# print(df.head(5))

## Deleting a Column
df = df.drop(columns=["Total"])
print(df.head(5))
