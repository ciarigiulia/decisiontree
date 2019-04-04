from math import floor
from time import time

import numpy as np
import pandas as pd
from docplex.mp.model import Model
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print('ciao ciao')
# to use wine dataset
df = pd.read_csv('wine.data.csv', header=None)
y = df[0]-1
df = df[df.columns[1:]]
data = df[df.columns[1:]].values
df2 = df
# to use fertility dataset
'''
df = pd.read_csv('fertility.csv', header=None)
y=df[9]
df=df[df.columns[0:9]]
data= df[df.columns[0:9]].values
df2=df

for i in range(len(y)):
    if y[i]=='N':
        y[i]=0
    else:
        y[i]=1
'''

# to use prova dataset
'''
df = pd.read_csv('prova.csv', header=None)
y = df[2]
df = df[df.columns[0:2]]
data = df[df.columns[0:2]].values
df2 = df
'''

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.05, random_state=1)

# DATA BETWEEN 0-1
scaler = MinMaxScaler()
df_scaled = scaler.fit(X_train)  # save object fitted only with train data
df_scaled = scaler.transform(X_train)
df = pd.DataFrame(df_scaled)  # scaled dataframe
df2 = scaler.transform(X_train)
df2 = pd.DataFrame(df2)

df_test = scaler.transform(X_test)  # apply same transformation to test set
for i in range(len(df_test)):
    for j in range(len(df_test[0])):
        if df_test[i][j] > 1:
            df_test[i][j] = 1
        elif df_test[i][j] < 0:
            df_test[i][j] = 0
df_test = pd.DataFrame(df_test)
df_test2 = scaler.transform(X_test)
df_test2 = pd.DataFrame(df_test2)

class tree(BaseEstimator):

    def __init__(self, depth=2, alpha=0.0, Nmin=0):

        self.depth = depth
        self.alpha = alpha
        self.Nmin = Nmin

        self.M2 = 1
        self.M1 = 1000
        self.M = 1000
        self.eps = []

        self.epsmin = 1e-5

        self.M1_test = 1000
        self.M_test = 1000
        self.eps_test = []

        self.Tb = []
        self.Tl = []
        self.floorTb = 0

        self.Al = {}
        self.Ar = {}
        self.parent = []

        self.classes = []
        self.features = []

        self.B = []
        self.l_test = []
        self.C = {}
        self.A = []

    def find_T(self, dataframe, y):

        T = pow(2, (self.depth + 1)) - 1  # nodes number
        self.floorTb = int(floor(T / 2))  # number of branch nodes
        self.Tb = np.arange(0, self.floorTb)  # range branch nodes
        self.Tl = np.arange(self.floorTb, T)  # range leaf nodes

        self.classes = np.unique(y.values)  # possible labels of classification
        self.features = dataframe.columns.values  # array of features
        return self.Tb, self.Tl, self.classes, self.features

    def find_eps(self, dataframe2):
        eps = np.zeros(len(dataframe2.columns))
        for i in range(0, len(dataframe2.columns)):

            if len(dataframe2) >= 2:
                vect = dataframe2[dataframe2.columns[i]].values
                newvect = dataframe2[dataframe2.columns[i]].values

                vect.sort()
                newvect.sort()
                vect = np.delete(vect, 0)
                newvect = np.delete(newvect, len(newvect) - 1)
                diff = vect - newvect
                diff2 = diff
                count = 0
                for j in range(len(diff)):
                    if diff[j] == 0:
                        diff2 = np.delete(diff2, j -count)
                        count +=1

                eps[i] = min(diff2)

            else:

                eps[i] = 1e-5

        return eps

    def find_epsmax(self, dataframe2):
        epsmax = max(self.eps)
        return epsmax

    def find_M1(self, dataframe2):
        M1 = 1 + self.find_epsmax(dataframe2)
        return M1

    def find_M(self, dataframe2):
        M = len(dataframe2)
        return M

    def find_pt(self):

        # define parent node function
        j = pow(2, self.depth - 1)
        for i in range(pow(2, self.depth) - pow(2, self.depth) + 1, pow(2, self.depth + 1) - 2 + 1, 2):
            self.parent.append(pow(2, self.depth - 1) - j)
            self.parent.append(pow(2, self.depth - 1) - j)
            j -= 1
        return self.parent

    def find_anc(self):

        for leaf in self.Tl:
            self.Al.update({leaf: []})
            self.Ar.update({leaf: []})

        for leaf in self.Tl:
            i = 1
            node = leaf  # nodo di cui vado a vedere il predecessore
            while i <= self.depth:  # ogni foglia ha un numero di predecessori pari alla profondità dell'albero
                pred = self.parent[node - 1]

                if node % 2 == 1:
                    self.Al.get(leaf).append(
                        pred)  # se il nodo è dispari lo metto nella lista della foglia del dizionario Al
                    node = pred  # il predecessore diventa il successivo nodo
                    i += 1

                elif node % 2 == 0:
                    self.Ar.get(leaf).append(pred)

                    i += 1
                    node = pred

        return self.Al, self.Ar

    def model(self, dataframe, dataframe2, y):

        self.eps = self.find_eps(dataframe2)
        self.M = self.find_M(dataframe2)
        self.M1 = self.find_M1(dataframe2)

        self.Tb, self.Tl, self.classes, self.features = self.find_T(dataframe, y)

        self.parent = self.find_pt()
        self.Al, self.Ar = self.find_anc()

        err = min(self.eps)

        # initialize the model

        mdl = Model(name='OCT_train')

        points = np.array(np.arange(0, len(dataframe)))  # array of point's indexes
        Num_points = len(dataframe)  # number of points

        # define Y matrix
        Y = np.arange(len(self.classes) * len(points)).reshape(len(points), len(self.classes))
        for i in range(0, len(points)):
            for k in range(0, len(self.classes)):
                if y.values[i] == k:
                    Y[i, k] = 1
                else:
                    Y[i, k] = -1

        # VARIABLES

        # for each branch node associate a feature, 1 if in node t I take feature f
        a = []
        for t in self.Tb:
            a.append(mdl.binary_var_list(len(self.features), ub=1, lb=0, name='a%d' % (t)))  # 'feature_in_node%d'%(t)

        # for each branch node associate a variable
        b = mdl.continuous_var_list(self.Tb, lb=0, name='b')  # 'hyperplane_coefficient'

        # per ogni nodo, indica se si applica lo split
        d = mdl.binary_var_list(self.Tb, name='d')  # node_with_split

        # per ogni nodo, è 1 se il punto si trova in quel nodo
        z = mdl.binary_var_matrix(Num_points, self.Tl, name='z')  # 'in_leaf_%d_pass_point_%d'

        l = mdl.binary_var_list(self.Tl, name='l')  # leaf_with_points

        c = mdl.binary_var_matrix(len(self.classes), self.Tl, name='c')  # class_of_leaf_%d_is_%d

        L = mdl.continuous_var_list(self.Tl, lb=0, name='L')  # loss_in_leaf

        Nt = mdl.continuous_var_list(self.Tl, name='Nt')  # points_in_leaf

        Nkt = mdl.continuous_var_matrix(len(self.classes), self.Tl, name='Nkt')  # points_in_leaf_%d_of_class_%d

        # CONSTRAINTS

        for le in range(len(self.Tl)):
            for k in range(len(self.classes)):
                mdl.add_constraint(L[le] >= Nt[le] - Nkt[k, le + self.floorTb] - self.M * (1 - c[k, le + self.floorTb]))

        for le in range(len(self.Tl)):
            for k in range(len(self.classes)):
                mdl.add_constraint(L[le] <= Nt[le] - Nkt[k, le + self.floorTb] + self.M * c[k, le + self.floorTb])

        for le in range(len(self.Tl)):
            for k in range(len(self.classes)):
                mdl.add_constraint(
                    Nkt[k, le + self.floorTb] == 0.5 * mdl.sum((1 + Y[i, k]) * z[i, le + self.floorTb] for i in points))

        for le in range(len(self.Tl)):
            mdl.add_constraint(Nt[le] == mdl.sum(z[p, le + self.floorTb] for p in points))

        for le in range(len(self.Tl)):
            mdl.add_constraint(l[le] == mdl.sum(c[k, le + self.floorTb] for k in range(len(self.classes))))

        for p in points:
            for le in range(len(self.Tl)):
                for n in self.Ar[le + self.floorTb]:
                    mdl.add_constraint(np.dot(dataframe.loc[p], a[n]) >= b[n] - self.M2 * (1 - z[p, le + self.floorTb]))

        for p in points:
            for le in range(len(self.Tl)):
                for m in self.Al[le + self.floorTb]:
                    mdl.add_constraint(
                        np.dot(dataframe.loc[p] + self.eps - err*np.ones(len(self.features)), a[m]) + err <= b[m] + self.M1 * (1 - z[p, le + self.floorTb]))

        for p in points:
            mdl.add_constraint(mdl.sum(z[p, le + self.floorTb] for le in range(len(self.Tl))) == 1)  #

        for le in range(len(self.Tl)):
            for p in points:
                mdl.add_constraint(z[p, le + self.floorTb] <= l[le])

        for le in range(len(self.Tl)):
            mdl.add_constraint(l[le] * self.Nmin <= mdl.sum(z[p, le + self.floorTb] for p in points))

        for t in self.Tb:
            mdl.add_constraint(d[t] == mdl.sum(a[t][f] for f in self.features))

        for t in self.Tb:
            mdl.add_constraint(b[t] <= d[t])
        for t in np.delete(self.Tb, 0):
            mdl.add_constraint(d[t] <= d[self.parent[t]])

        for i in range(0, len(self.Tl), 2):
            mdl.add_constraint(l[i] <= d[self.find_pt()[i + self.depth]])

        mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * mdl.sum(d[t] for t in self.Tb))
        mdl.print_information()
        
        start = time()

        # mdl.set_time_limit(600)
        # mdl.parameters.mip.tolerances.mipgap(0.1)
        mdl.solve(log_output=True)
        mdl.report()
        print(mdl.solve_details)
        # mdl.print_solution()

        fit_time = time() - start
        print('time to solve the model:', fit_time)
        return mdl

    def fit(self, dataframe, dataframe2, y):

        # OBJECTIVE FUNCTION

        # per calcolare il train error
        sol = self.model(dataframe, dataframe2, y)

        train_error = 0
        for leaf in self.Tl:
            train_error += sol.solution.get_value('L_' + str(leaf))
        train_error = train_error / self.M
        print('train_error:', train_error)

        for t in self.Tb:
            self.B.append(sol.solution.get_value('b_' + str(t)))

        for leaf in self.Tl:
            self.l_test.append(sol.solution.get_value('l_' + str(leaf)))

        for k in range(len(self.classes)):
            for leaf in self.Tl:
                self.C.update({(k, leaf): sol.solution.get_value('c_' + str(k) + '_' + str(leaf))})

        for t in self.Tb:
            A_list = []
            for f in self.features:
                A_list.append(sol.solution.get_value('a' + str(t) + '_' + str(f)))
            self.A.append(A_list)

        return sol

    def test_model(self, dataframe, dataframe2, y):
        self.eps_test = self.find_eps(dataframe2)
        self.M1_test = self.find_M1(dataframe2)

        err_test = min(self.eps_test)

        Num_points = len(dataframe)
        points = np.array(np.arange(0, len(dataframe)))  # array of point's indexes

        mdl = Model('OCT_test')

        z = mdl.binary_var_matrix(Num_points, self.Tl, name='z')

        # CONSTRAINTS

        for p in points:
            for le in range(len(self.Tl)):
                for n in self.Ar[le + self.floorTb]:
                    mdl.add_constraint(
                        np.dot(dataframe.loc[p], self.A[n]) >= self.B[n] - self.M2 * (1 - z[p, le + self.floorTb]))

        for p in points:
            for le in range(len(self.Tl)):
                for m in self.Al[le + self.floorTb]:

                    mdl.add_constraint(np.dot(dataframe.loc[p] + self.eps_test - err_test*np.ones(len(self.features)), self.A[m]) + err_test <= self.B[m] + self.M1_test * (
                            1 - z[p, le + self.floorTb]))

        for p in points:
            mdl.add_constraint(
                mdl.sum(z[p, le + self.floorTb] for le in range(len(self.Tl))) == 1)  # each poin associate to a leaf

        for le in range(len(self.Tl)):
            for p in points:
                mdl.add_constraint(z[p, le + self.floorTb] <= self.l_test[le])

        mdl.print_information()

        start_test = time()



        mdl.solve(log_output=True)
        print(mdl.solve_details)
        mdl.report()
        predict_time = time() - start_test
        print('Time to predict:', predict_time)

        test_error = 0
        for p in points:
            for leaf in self.Tl:
                if mdl.solution.get_value(z[p, leaf]) == 1:
                    if self.C[y.values[p], leaf] == 1:

                        test_error += 0
                    elif self.C[y.values[p], leaf] == 0:

                        test_error += 1

        test_error = test_error / len(dataframe)
        print('test_error:', test_error)

        return mdl


t = tree(depth=2, alpha=0.5, Nmin=1)
f = t.fit(df, df2, y_train)
predict = t.test_model(df_test, df_test2, y_test)
