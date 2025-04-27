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
scale_nums = False

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

plt.show()


for lang in range(2):
    features = data.iloc[1:, [5, 8, 10, 11, 14]].values.astype(int)
    features = np.array(list(filter(lambda x: x[-1] == lang, features)))

    features =  np.array(list(zip(*list(zip(*features))[:-2])))

    cluster_count = 2 if lang == 1 else 3

    if (not standardized) or scale_nums:
        features = list(zip(*features))
        for i in range(len(features)):
            features[i] /= max(features[i])

        features = np.array(list(zip(*features)))

    kmeans = KMeans(n_clusters=cluster_count)
    kmeans.fit(features)

    values = np.zeros((cluster_count, 3))
    count = np.zeros((cluster_count, 1))


    for i in zip(features, kmeans.labels_):
        values[i[1]] += i[0]
        count[i[1]] += 1

    cluster_averages = values / count



    #print(cluster_averages)

    #plt.figure(lang + 1)

    lang_name = "English" if lang == 1 else "Hebrew"
    line_type = "dashed" if lang == 1 else "solid"

    for i in range(cluster_count):
        plt.plot(range(3), cluster_averages[i],
                 marker='o', label=f"{lang_name} {int(count[i] / sum(count) * 1000) / 10}%", linestyle=line_type)

column_names = data.iloc[0, [5, 8, 10]]

plt.xticks(range(3), column_names)
plt.yticks(range(4))
plt.legend()
#plt.title(f"Song Popularity Clusters")

plt.show()
"""
for lang in range(2):
    lang_name = "English" if lang == 1 else "Hebrew"

    features = data.iloc[1:, [2, 8, 14]].values.astype(int)
    features = np.array(list(filter(lambda x: x[-1] == lang, features)))

    features = np.array(list(zip(*list(zip(*features))[:-1])))

    rank, streams = np.array(list(zip(*features)))

    plt.scatter(rank, streams, label=lang_name)

plt.legend()
plt.xlabel("Rank")
plt.ylabel("Streams")

plt.show()