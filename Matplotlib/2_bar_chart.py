from matplotlib import pyplot as plt
import numpy as np

# Ages for x-axis
ages_x = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
# Hacky Solution to show multiple bar chart
x_indexes = np.arange(len(ages_x))
bar_width = 0.25

# Median Python Developer Salaries by Age
py_dev_y = [20046, 17100, 20000, 24744, 30500, 37732, 41247, 45372, 48876, 53850, 57287, 63016, 65998, 70003, 70000, 71496, 75370, 83640, 84666]
# Bar Chart Plot
plt.bar(x_indexes - bar_width, py_dev_y, width = bar_width, label ='Python Devs')

# Median JavaScript Developer Salaries by Age
js_dev_y = [16446, 16791, 18942, 21780, 25704, 29000, 34372, 37810, 43515, 46823, 49293, 53437, 56373, 62375, 66674, 68745, 68746, 74583, 79000]
plt.bar(x_indexes, js_dev_y, width = bar_width, label ='Javascript')


# Median Developer Salaries by Age
dev_y = [17784, 16500, 18012, 20628, 25206, 30252, 34368, 38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928, 67317, 68748, 73752, 77232]
plt.bar(x_indexes + bar_width, dev_y, width = bar_width,  label ='All Devs')

# Good way to do it! Label each plot, then legend can be ran empty
plt.legend()

# Uses x_indexes for tickets and ages_x for the label
plt.xticks(ticks=x_indexes, labels=ages_x)

# Labeling the Graphs
plt.title('Median Salary (USD) By Age')
plt.xlabel('Age')
plt.ylabel('Income')



# Add padding to graph 
plt.tight_layout()

# Show graph
plt.show()