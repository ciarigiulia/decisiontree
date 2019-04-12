import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import graphviz

df = pd.read_csv('wine.data.csv', header=None)
clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, max_features=1, random_state=1)
y = df[0]-1
df = df[df.columns[1:]]
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0, random_state=1)

scaler = MinMaxScaler()
df_scaled = scaler.fit(X_train)  # save object fitted only with train data
df_scaled = scaler.transform(X_train)
df = pd.DataFrame(df_scaled)  # scaled dataframe

'''df_test = scaler.transform(X_test)  # apply same transformation to test set

for i in range(len(df_test)):
    for j in range(len(df_test[0])):
        if df_test[i][j] > 1:
            df_test[i][j] = 1
        elif df_test[i][j] < 0:
            df_test[i][j] = 0
df_test = pd.DataFrame(df_test)'''

f = clf.fit(df, y_train)
print(clf.score(df, y_train))
print(clf.apply(df))
#print(clf.decision_path(df))
#print(clf.apply(df_test))
#print(clf.decision_path(df_test))
#p = clf.predict(df_test)
#print(clf.score(df_test,y_test))

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

print(clf.tree_.value)

n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold
print(n_nodes)
print(children_left)
print(children_right)
print(feature)
print(threshold)

#dot_data = tree.export_graphviz(clf, out_file=None, class_names=['0', '1', '2'])
#graph = graphviz.Source(dot_data)
#graph.render(filename="wine_full_sklearn_D3", directory='/Users/giuliaciarimboli/Desktop/laurea magistrale/classification trees/graphs', view=True)


