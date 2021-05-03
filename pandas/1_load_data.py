import pandas as pd

## Read  CSV
df = pd.read_csv("panda/pokemon_data.csv")

## Read XLSX
df_xlsx = pd.read_excel("panda/pokemon_data.xslx")

## Read a txt file
df_txt = pd.read_csv("panda/pokemon_data.txt", delimiter="\t")
