import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from docplex.mp.model import Model
import numpy as np
from math import floor
from docplex.mp.solution import SolveSolution
from sklearn import tree
import graphviz

# import data
df = pd.read_csv('wine.data.csv', header=None)
y = df[0]-1
df = df[df.columns[1:]]
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0, random_state=1)

# scale data between 0-1
scaler = MinMaxScaler()
df_scaled = scaler.fit(X_train)  # save object fitted only with train data
df_scaled = scaler.transform(X_train)
df = pd.DataFrame(df_scaled)  # scaled dataframe

# construct the tree with sklearn
depth = 2
clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=10, max_features=1, random_state=1)
f = clf.fit(df, y_train)

# dot_data = tree.export_graphviz(clf, out_file=None, class_names=['0', '1', '2'])
# graph = graphviz.Source(dot_data)
# graph.render(filename="wine_full_sklearn_D3_", directory='/Users/giuliaciarimboli/Desktop/laurea magistrale/classification trees/graphs', view=True)

sk_features = clf.tree_.feature
sk_b = clf.tree_.threshold
sk_val = clf.tree_.value
sk_z = clf.apply(df)

T = pow(2, (depth + 1)) - 1  # nodes number
floorTb = int(floor(T / 2))  # number of branch nodes
Tb = np.arange(0, floorTb)  # range branch nodes
Tl = np.arange(floorTb, T)  # range leaf nodes
nodes = np.append(Tb,Tl)

classes = np.unique(y.values)  # possible labels of classification
features = df.columns.values  # array of features
Num_points=len(df)

mdl = Model(name='OCT - warm start')
a = []
for t in Tb:
    a.append(mdl.binary_var_list(len(features), ub=1, lb=0, name='a%d' % (t)))  # 'feature_in_node%d'%(t)
# for each branch node associate a variable
b = mdl.continuous_var_list(Tb, lb=0, name='b')  # 'hyperplane_coefficient'

# per ogni nodo, indica se si applica lo split
d = mdl.binary_var_list(Tb, name='d')  # node_with_split

# per ogni nodo, è 1 se il punto si trova in quel nodo
z = mdl.binary_var_matrix(Num_points, Tl, name='z')  # 'in_leaf_%d_pass_point_%d'

l = mdl.binary_var_list(Tl, name='l')  # leaf_with_points

c = mdl.binary_var_matrix(len(classes), Tl, name='c')  # class_of_leaf_%d_is_%d

L = mdl.continuous_var_list(Tl, lb=0, name='L')  # loss_in_leaf

Nt = mdl.continuous_var_list(Tl, name='Nt')  # points_in_leaf

Nkt = mdl.continuous_var_matrix(len(classes)+1, Tl, name='Nkt')  # points_in_leaf_%d_of_class_%d
if depth==2:
    idx_sk = [0,1,4,2,3,5,6]
#   nodes =  [0,1,2,3,4,5,6]
elif depth==3:
    idx_sk =[0,1,8,2,5,9,12,3,4,6,7 ,10,11,13,14]
#   nodes  =[0,1,2,3,4,5, 6,7,8,9,10,11,12,13,14]

m = SolveSolution(mdl)
count = 0
j = -1
for node in idx_sk:
    j += 1
    if sk_features[j] >= 0:
        i = list(idx_sk).index(j) # prendo l'indice j-esimo della lista dei nodi di sklearn, equivalente al nodo oct
        feat = sk_features[j] # è la feature da prendere nell'i esimo nodo
        m.add_var_value('a%d_%d' % (i, feat),1)
        m.add_var_value(('b_%d'%(i)), sk_b[j])
        count += 1
for t in Tb:
    m.add_var_value(('d_%d'%(t)), 1)
for leaf in Tl:
    m.add_var_value(('l_%d'%(leaf)), 1)

jj=-1
for node in idx_sk:
    jj+=1
    k = np.argmax(sk_val[jj][0])
    num = np.sum(sk_val[jj][0])
    ii = list(idx_sk).index(jj)
    if ii in Tl:
        m.add_var_value('c_%d_%d'%(k,ii), 1)
        m.add_var_value('Nt_%d'%(ii), num)
        for kl in range(len(classes)):
            m.add_var_value('Nkt_%d_%d'%(kl,ii), sk_val[jj][0][kl])
for data in range(Num_points):
    foglia = list(idx_sk).index(sk_z[data])
    m.add_var_value('z_%d_%d'%(data,foglia), 1)


for i in Tb:
    for f in features:
        if m.get_value('a%d_%d'%(i,f))==1:
            print('la feature del nodo %d è la %d:'%(i,f))
for i in Tb:
    print('il valore di b nel nodo %d è pari a:'%(i),m.get_value('b_%d'%(i)))
for leaf in Tl:
    for kl in range(len(classes)):
        print(kl, leaf, m.get_value('Nkt_%d_%d'%(kl,leaf)))

for leaf in Tl:
    for kl in range(len(classes)):
        if m.get_value(('c_%d_%d'%(kl,leaf)))==1:
             print('la classe nella foglia %d è la %d'%(leaf,kl))
for da in range(Num_points):
    for leaf in Tl:

        if m.get_value('z_%d_%d'%(da,leaf))==1:
            print('il dato %d è nella foglia %d'%(da,leaf))

for leaf in Tl:
    print('nella foglia %d ci sono:'%(leaf),m.get_value('Nt_%d'%(leaf)), 'dati')
mdl.print_information()
print(m)