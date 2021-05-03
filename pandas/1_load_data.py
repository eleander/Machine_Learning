import pandas as pd

## Read  CSV
df = pd.read_csv("pandas/pokemon_data.csv")

## Read XLSX
df_xlsx = pd.read_excel("pandas/pokemon_data.xlsx")

## Read a txt file
df_txt = pd.read_csv("pandas/pokemon_data.txt", delimiter="\t")
