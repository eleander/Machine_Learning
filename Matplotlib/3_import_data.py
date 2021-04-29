import csv
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt

# Standard Library Way
# with open('Matplotlib/data.csv') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     language_counter = Counter()

#     # Reads only first item
#     # row = next(csv_reader)
#     # print(row['LanguagesWorkedWith'].split(';'))

#     for row in csv_reader:
#         language_counter.update(row['LanguagesWorkedWith'].split(';'))


# Panda Way
data = pd.read_csv('Matplotlib/data.csv')
ids = data['Responder_id']
lang_responses = data['LanguagesWorkedWith']

language_counter = Counter()

for response in lang_responses:
    language_counter.update(response.split(';'))

# Prints from most to least popular programming language
# print(language_counter)

# Prints 15 most common languages
# In form [('language_name', num)]
# print(language_counter.most_common(15))


# Organise the language_counter into 2 lists
languages = []
popularity = []

for item in language_counter.most_common(15):
    languages.append(item[0])
    popularity.append(item[1])
    
# print(languages)
# print(popularity)

# Reverses both arrays
languages.reverse()
popularity.reverse()

# Horizontal Bar Chart
plt.barh(languages,popularity)

# Labeling the Graphs
plt.title('Most Popular Languages')
plt.xlabel('Number of People Who Use')

plt.tight_layout()

# Show graph
plt.show()