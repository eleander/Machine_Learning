import pandas as pd

df = pd.read_csv("panda/pokemon_data.csv")

## Read headers
# print(df.columns)

## Read each column
print(df["Name"])

## Read 5 of a specific column
print(df["Name"][0:5])

## Read multiple columns
print(df[["Name", "Defense"]][0:5])

## Read each row
print(df.head(4))

# Read only rows 7 to 10 (not inclusive)
print(df.iloc[7:10])

## Read a specific column
print(df.iloc[2, 1])

# Itterate through rows
for index, row in df.iterrows():
    print(index, row["Name"])

# Finding specific data
print(df.loc[df["Type 1"] == "Fire"])
