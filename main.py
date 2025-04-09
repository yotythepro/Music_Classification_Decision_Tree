from io import StringIO

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
import pydotplus

standardized = True

dataset_file_name = "song_Data.class.csv" if standardized else "song_Data_Non_Standard.csv"
image_name = "tree_standard.png" if standardized else "tree_non_standard.png"

data = pd.read_csv(dataset_file_name, header=None)

X = data.iloc[1:, [5, 7, 8]].values
Y = data.iloc[1:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train = X_train.astype(int)
Y_train = Y_train.astype(int)

clf = DecisionTreeClassifier(criterion="entropy",  min_impurity_decrease=0.01)
clf.fit(X_train, Y_train)

X_test = X_test.astype(int)
Y_test = Y_test.astype(int)
Y_pred = clf.predict(X_test)

print(metrics.accuracy_score(Y_test, Y_pred))

print(metrics.confusion_matrix(Y_test == 1, Y_pred == 1))
(TN, FN), (FP, TP) = metrics.confusion_matrix(Y_test == 1, Y_pred == 1)
print(f"Sensitivity: {TP / (TP + FN)}")
print(f"Specificity: {TN / (TN + FP)}")

df_predictors_names = list(data.iloc[0, [5, 7, 8]])
df_class_names = ["Hebrew", "English"]

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=df_predictors_names, class_names=df_class_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png(image_name)
