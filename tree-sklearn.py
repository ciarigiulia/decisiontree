import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import graphviz
import numpy as np
datasets = ['acute-inflammations-1.csv', 'fertility.csv', 'parkinsons.csv', 'spectf80.csv',
            'connectionist-bench-sonar.csv', 'ionosphere.csv', 'ThoracicSurgery.csv', 'climate-model-crashes.csv',
            'banknote-authentication.csv', 'seismic-bumps.csv']
'''for data in datasets:
    df = pd.read_csv(data, header=None)
    clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1, random_state=1)
    y = df[0]
    df = df[df.columns[1:]]
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=1)

    scaler = MinMaxScaler()
    df_scaled = scaler.fit(X_train)  # save object fitted only with train data
    df_scaled = scaler.transform(X_train)
    df = pd.DataFrame(df_scaled)  # scaled dataframe

    df_test = scaler.transform(X_test)  # apply same transformation to test set

    for i in range(len(df_test)):
        for j in range(len(df_test[0])):
            if df_test[i][j] > 1:
                df_test[i][j] = 1
            elif df_test[i][j] < 0:
                df_test[i][j] = 0
    df_test = pd.DataFrame(df_test)'''

def divide_train_and_test(X, y, splitPercentage, stratify=None):
    if stratify == None:
        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=1 - splitPercentage, random_state=42,
                                                        stratify=stratify)
    else:
        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=1 - splitPercentage, random_state=42,
                                                        stratify=y)
    return xTrain, yTrain, xTest, yTest


def from_regression_to_class(yRegr, bins, labels):
    yClass = yRegr.reshape(-1, )
    y = np.asarray(pd.cut(yClass, bins=bins, labels=labels))
    y = y.reshape(-1, 1)
    return y


np.random.seed(8)

df = pd.read_excel('Neurologici_Nuovi.xlsx')

y = df["COD_80"]
y = y.values.reshape(-1, 1)
X = df.drop(columns=["COD_80"])
X = X.values
threshold = 80
print(np.sum(y >= threshold))
print(np.sum(y < threshold))

bins = [0, threshold, 101]
labels = [0, 1]
y = from_regression_to_class(y, bins, labels)

splitPercentage = 0.7
xTrain, yTrain, xTest, yTest = divide_train_and_test(X, y, splitPercentage, stratify=None)
xTrain = pd.DataFrame(xTrain)
xTest = pd.DataFrame(xTest)
scaler = MinMaxScaler()
df_scaled = scaler.fit(xTrain)
df = scaler.transform(xTrain)
df = pd.DataFrame(df)
df2 = scaler.transform(xTrain)
df2 = pd.DataFrame(df2)
df_test = scaler.transform(xTest)  # apply same transformation to test set
yTest = pd.DataFrame(yTest)
yTrain = pd.DataFrame(yTrain)

for i in range(len(df_test)):
    for j in range(len(df_test[0])):
        if df_test[i][j] > 1:
            df_test[i][j] = 1
        elif df_test[i][j] < 0:
            df_test[i][j] = 0

df_test = pd.DataFrame(df_test)
clf = DecisionTreeClassifier(max_depth=1, min_samples_leaf=20, random_state=1)

f = clf.fit(df, yTrain)
print('train error', 1-clf.score(df, yTrain))
print('test error', 1-clf.score(df_test,yTest))

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render(filename="prova", directory='/Users/giuliaciarimboli/Desktop/laurea magistrale/classification trees/graphs', view=True)
# The decision estimator has an attribute called tree_  which stores the entire
# tree structure and allows access to low level attributes. The binary tree
# tree_ is represented as a number of parallel arrays. The i-th element of each
# array holds information about the node `i`. Node 0 is the tree's root. NOTE:
# Some of the arrays only apply to either leaves or split nodes, resp. In this
# case the values of nodes of the other type are arbitrary!
#
# Among those arrays, we have:
#   - left_child, id of the left child of the node
#   - right_child, id of the right child of the node
#   - feature, feature used for splitting the node
#   - threshold, threshold value at the node
#


'''n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold
#print(n_nodes)
#print(children_left)
#print(children_right)
print(feature)
print(threshold)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render(filename="prova", directory='/Users/giuliaciarimboli/Desktop/laurea magistrale/classification trees/graphs', view=True)
print(clf.tree_.children_left)
print(clf.tree_.children_right)
idx=[0]
j=1
left = clf.tree_.children_left
right = clf.tree_.children_right
for i in range(len(clf.tree_.children_left)):
    if idx[i]>=0:
        node = idx[i]
    if clf.tree_.children_left[node]>0:
        idx.insert(j, clf.tree_.children_left[node])
        j+=1
    if clf.tree_.children_right[node]>0:
        idx.insert(j, clf.tree_.children_right[node])
        j+=1'''