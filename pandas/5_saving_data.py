import pandas as pd

df = pd.read_csv("pandas/pokemon_data.csv")

## Add a new column
df["Total"] = (
    df["HP"]
    + df["Attack"]
    + df["Defense"]
    + df["Sp. Atk"]
    + df["Sp. Def"]
    + df["Speed"]
)

## Get all column values
# cols = list(df.columns)

## Change where total is
# df = df[cols[0:4] + [cols[-1]] + cols[4:12]]
# print(df.head(5))

## Save as CSV
# df.to_csv("panda/modified.csv", index=False)

## Save as Excel
# df.to_excel("modified.xlsx", index=False)

## Save as Json
# df.to_json("modified.json", index=False)


## Convert to CSV from Txt
# df.to_csv("modified.txt", index=False, sep="\t")
