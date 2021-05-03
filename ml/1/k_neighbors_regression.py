# Code example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Very easy to change which learning algorithm is used
# import sklearn.linear_model
import sklearn.neighbors
from prepare_data import prepare_country_stats

# Load the data
datapath = "ml/1/"
oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=",")
gdp_per_capita = pd.read_csv(
    datapath + "gdp_per_capita.csv",
    thousands=",",
    delimiter="\t",
    encoding="latin1",
    na_values="n/a",
)

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
# Note: Plotting a dataframe
country_stats.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")
plt.show()

# Very easy to change which model is used!
# Select a K Neighbours Regressor model
# model = sklearn.linear_model.LinearRegression()
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new))  # outputs [[5.76666667]]
