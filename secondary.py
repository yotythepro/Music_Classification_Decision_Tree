#from io import StringIO

#from sklearn.model_selection import train_test_split
import pandas as pd
#from sklearn.tree import DecisionTreeClassifier
#from sklearn import tree
#from sklearn import metrics
#import pydotplus
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.core import shape
from sklearn.cluster import KMeans

standardized = False

dataset_file_name = "song_Data.class.csv" if standardized else "song_Data_Non_Standard.csv"
#image_name = "corr_standard.png" if standardized else "corr_non_standard.png"

data = pd.read_csv(dataset_file_name, header=None)
"""
correlation = data.iloc[1:, [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].corr()
column_names = data.iloc[0, [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]

axis_corr = sns.heatmap(
correlation,
vmin=-1, vmax=1, center=0,
cmap=sns.diverging_palette(50, 500, n=500),
square=True, xticklabels=column_names, yticklabels=column_names
)

plt.show()

print(data.iloc[1:, 10].values.astype(int))

plt.scatter(data.iloc[1:, 13].values.astype(int), data.iloc[1:, 8].values.astype(int))
plt.plot(range(30), range(30))

plt.show()"""

features = data.iloc[1:, [5, 8, 10, 11, 14]].values.astype(int)

features = list(zip(*features))
for i in range(len(features)):
    features[i] /= max(features[i])

features = np.array(list(zip(*features)))

kmeans = KMeans(n_clusters=3)
kmeans.fit(features)

values = np.zeros((3, 5))
count = np.zeros((3, 1))


for i in zip(features, kmeans.labels_):
    values[i[1]] += i[0]
    count[i[1]] += 1

cluster_averages = values / count

"""cluster_averages = np.array([[0.56185567, 1.01546392, 1., 1., 0.],
 [2.48812665, 1.25593668, 1.20580475, 1.35883905, 0.32717678],
 [0.43877551, 1.00510204, 1., 1.01530612, 1.]])"""

column_names = data.iloc[0, [5, 8, 10, 11, 14]]

print(cluster_averages)

for i in range(3):
    plt.plot(range(5), cluster_averages[i])
plt.show()