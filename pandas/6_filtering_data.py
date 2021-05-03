import pandas as pd

df = pd.read_csv("pandas/pokemon_data.csv")

category = "Type 1"

## Finding specific data
print(df.loc[df[category] == "Grass"])

## Finding multiple specific data
## And
new_df = df.loc[
    (df[category] == "Grass") & (df[category] == "Poison") & (df["HP"] < 70)
]
print(new_df)

## Finding multiple specific data
## Or
new_df = df.loc[(df[category] == "Grass") | (df[category] == "Poison")]
print(new_df)


## Save new dataframe
# new_df.reset_index(drop=True, inplace=True)
# new_df.to_csv("filter.csv")
