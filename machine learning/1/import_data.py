import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

datapath = "ml/1/"
oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=",")
gdp_per_capita = pd.read_csv(
    datapath + "gdp_per_capita.csv",
    thousands=",",
    delimiter="\t",
    encoding="latin1",
    na_values="n/a",
)

print(oecd_bli.head(4))
print(gdp_per_capita.head(4))
