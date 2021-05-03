import pandas as pd

df = pd.read_csv("panda/pokemon_data.csv")

## Gives cool info
# print(df.describe())

## Sort name in ascending order
# print(df.sort_values("Name"))

## Sort name in descending order
# print(df.sort_values("Name", ascending=False))

## Sort for Type 1 and HP
# Type 1 = ascending
# HP = descending
print(df.sort_values(["Type 1", "HP"], ascending=[1, 0]))
