from sklearn.model_selection import train_test_split

from math import ceil
from math import floor

import numpy as np
import pandas as pd
import pygraphviz as pgv
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# import graphviz
import csv


class OptimalTree(BaseEstimator):

    def __init__(self, depth=2, alpha=0.0, Nmin=1, max_features=4, version='multivariate', name='OCT-H',
                 dataset='wine.data.csv', mipstart='CART', relax=0):

        self.depth = depth
        self.alpha = alpha
        self.Nmin = Nmin
        self.name = name
        self.dataset = dataset
        self.version = version
        self.relax = relax

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

        self.mipstart = mipstart
        self.M2 = 1
        self.M1 = 1000
        self.M = 1000
        self.eps = []
        self.epsmin = 1e-3

        self.mu = 0.005
        self.bigM = 2
        self.max_features = max_features

        self.csvrow = [0] * 31

    def find_T(self, dataframe, y):

        T = pow(2, (self.depth + 1)) - 1  # nodes number
        self.floorTb = int(floor(T / 2))  # number of branch nodes
        Tb = np.arange(0, self.floorTb)  # range branch nodes
        Tl = np.arange(self.floorTb, T)  # range leaf nodes

        classes = np.unique(y.values)  # possible labels of classification
        features = dataframe.columns.values  # array of features
        return Tb, Tl, classes, features

    def find_pt(self):

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
                        diff2 = np.delete(diff2, j - count)
                        count += 1
                if len(diff2) > 0 and min(diff2) >= 1e-6:
                    eps[i] = min(diff2)
                else:
                    eps[i] = 1e-5
            else:

                eps[i] = 1e-5
        print(eps)
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

    def model(self, dataframe, dataframe2, y):

        np.random.seed(1)
        self.Tb, self.Tl, self.classes, self.features = self.find_T(dataframe, y)
        self.parent = self.find_pt()
        self.Al, self.Ar = self.find_anc()
        points = dataframe.index
        Num_points = len(dataframe)  # number of points

        if self.relax == 0:
            if self.version == 'univariate':
                self.eps = self.find_eps(dataframe2)
                self.M = self.find_M(dataframe2)
                self.M1 = self.find_M1(dataframe2)
                err = min(self.eps)

            elif self.version == 'multivariate':
                self.M = len(dataframe)

            mdl = Model(name=self.name)
            mdl.clear()

            Y = np.arange(len(self.classes) * len(points)).reshape(len(points), len(self.classes))
            for i in range(0, len(points)):
                for k in range(0, len(self.classes)):
                    k1 = self.classes[k]
                    if y.values[i] == k1:
                        Y[i, k] = 1
                    else:
                        Y[i, k] = -1
            # VARIABLES

            if self.version == 'univariate':
                a = []
                for t in self.Tb:
                    a.append(mdl.binary_var_list(len(self.features), name='a%d' % t))
                b = mdl.continuous_var_list(self.Tb, lb=0,
                                            name='b')
            elif self.version == 'multivariate':
                a = []
                for t in self.Tb:
                    a.append(mdl.continuous_var_list(len(self.features), ub=1, lb=-1, name='a%d' % t))
                a_hat = []
                for t in self.Tb:
                    a_hat.append(mdl.continuous_var_list(len(self.features), ub=1, lb=-1,
                                                         name='a_hat%d' % t))
                if self.name == 'OCT-H' or self.name == 'S1' or self.name == 'St':
                    s = []
                    for t in self.Tb:
                        s.append(mdl.binary_var_list(len(self.features), name='s%d' % t))
                b = mdl.continuous_var_list(self.Tb, lb=-1, ub=1,
                                            name='b')  # TODO verificare se anche nel modello multivariate il lower bound di b è zero e eventualmente metterlo

            d = mdl.binary_var_list(self.Tb, name='d')

            z = mdl.binary_var_matrix(Num_points, self.Tl, name='z')

            l = mdl.binary_var_list(self.Tl, name='l')

            c = mdl.binary_var_matrix(len(self.classes), self.Tl, name='c')

            L = mdl.continuous_var_list(self.Tl, lb=0, name='L')

            Nt = mdl.continuous_var_list(self.Tl, name='Nt')

            Nkt = mdl.continuous_var_matrix(len(self.classes), self.Tl, name='Nkt')
            # CONSTRAINTS
            if self.version == 'multivariate':
                # mdl.add_constraint(d[0] == 1)
                for le in range(len(self.Tl)):
                    for k in range(len(self.classes)):
                        mdl.add_constraint(
                            L[le] >= Nt[le] - Nkt[k, le + self.floorTb] - self.M * (1 - c[k, le + self.floorTb]))

                for le in range(len(self.Tl)):
                    for k in range(len(self.classes)):
                        mdl.add_constraint(
                            L[le] <= Nt[le] - Nkt[k, le + self.floorTb] + self.M * c[k, le + self.floorTb])

                for le in range(len(self.Tl)):
                    for k in range(len(self.classes)):
                        mdl.add_constraint(
                            Nkt[k, le + self.floorTb] == 0.5 * mdl.sum(
                                (1 + Y[i, k]) * z[i, le + self.floorTb] for i in range(len(points))))

                for le in range(len(self.Tl)):
                    mdl.add_constraint(Nt[le] == mdl.sum(z[p, le + self.floorTb] for p in range(len(points))))

                for le in range(len(self.Tl)):
                    mdl.add_constraint(l[le] == mdl.sum(c[k, le + self.floorTb] for k in range(len(self.classes))))

                for p in range(len(points)):
                    mdl.add_constraint(mdl.sum(z[p, le + self.floorTb] for le in range(len(self.Tl))) == 1)

                for le in range(len(self.Tl)):
                    for p in range(len(points)):
                        mdl.add_constraint(z[p, le + self.floorTb] <= l[le])

                for le in range(len(self.Tl)):
                    mdl.add_constraint(
                        l[le] * self.Nmin <= mdl.sum(z[p, le + self.floorTb] for p in range(len(points))))

                for t in np.delete(self.Tb, 0):
                    mdl.add_constraint(d[t] <= d[self.parent[t]])

                for p in range(len(points)):
                    p1 = points[p]
                    for le in range(len(self.Tl)):
                        for n in self.Ar[le + self.floorTb]:
                            mdl.add_constraint(
                                np.dot(dataframe.loc[p1], a[n]) >= b[n] - self.bigM * (1 - z[p, le + self.floorTb]))

                for p in range(len(points)):
                    p1 = points[p]
                    for le in range(len(self.Tl)):
                        for m in self.Al[le + self.floorTb]:
                            mdl.add_constraint(
                                np.dot(dataframe.loc[p1], a[m]) + self.mu <= b[m] + (self.bigM + self.mu) * (
                                        1 - z[p, le + self.floorTb]))
                for t in self.Tb:
                    mdl.add_constraint(d[t] >= mdl.sum(a_hat[t][f] for f in self.features))

                for t in self.Tb:
                    for f in self.features:
                        mdl.add_constraint(a_hat[t][f] >= a[t][f])

                for t in self.Tb:
                    for f in self.features:
                        mdl.add_constraint(a_hat[t][f] >= -a[t][f])
                if self.name == 'OCT-H' or self.name == 'S1' or self.name == 'St':

                    for t in self.Tb:
                        for f in self.features:
                            mdl.add_constraint(-s[t][f] <= a[t][f])

                    for t in self.Tb:
                        for f in self.features:
                            mdl.add_constraint(a[t][f] <= s[t][f])

                    for t in self.Tb:
                        for f in self.features:
                            mdl.add_constraint(s[t][f] <= d[t])

                    for t in self.Tb:
                        mdl.add_constraint(mdl.sum(s[t][f] for f in self.features) >= d[t])

                for t in self.Tb:
                    mdl.add_constraint(b[t] <= d[t])

                for t in self.Tb:
                    mdl.add_constraint(b[t] >= -d[t])
    
                    if len(self.classes) <= len(
                            self.Tl):
                        for k in range(len(self.classes)):
                            mdl.add_constraint(1 <= mdl.sum(c[k, le + self.floorTb] for le in range(len(self.Tl))))

                if self.name == 'St':
                    for t in self.Tb:
                        mdl.add_constraint(mdl.sum(s[t][f] for f in self.features) <= self.max_features)
                if self.name == 'S1' or self.name == 'St':
                    mdl.add_constraint(
                        mdl.sum(s[t][f] for t in self.Tb for f in self.features) <= self.max_features * len(self.Tb))

            '''elif self.version == 'univariate':
                for p in range(len(points)):
                    p1 = points[p]
                    for le in range(len(self.Tl)):
                        for n in self.Ar[le + self.floorTb]:
                            mdl.add_constraint(
                                np.dot(dataframe.loc[p1], a[n]) >= b[n] - self.M2 * (1 - z[p, le + self.floorTb]))

                for p in range(len(points)):
                    p1 = points[p]
                    for le in range(len(self.Tl)):
                        for m in self.Al[le + self.floorTb]:
                            mdl.add_constraint(
                                np.dot(dataframe.loc[p1] + self.eps - err * np.ones(len(self.features)), a[m]) + err <= b[
                                    m] + self.M1 * (1 - z[p, le + self.floorTb]))
                for t in self.Tb:
                    mdl.add_constraint(d[t] == mdl.sum(a[t][f] for f in self.features))
                for t in self.Tb:
                    mdl.add_constraint(b[t] <= d[t])

                for i in range(0, len(self.Tl), 2):
                    mdl.add_constraint(l[i] <= d[self.find_pt()[i + self.depth]])'''
            if self.version == 'univariate':
                Y = np.arange(len(self.classes) * len(points)).reshape(len(points), len(self.classes))
                for i in range(0, len(points)):
                    for k in range(0, len(self.classes)):
                        if y.values[i] == k:
                            Y[i, k] = 1
                        else:
                            Y[i, k] = -1
                for le in range(len(self.Tl)):
                    for k in range(len(self.classes)):
                        mdl.add_constraint(
                            L[le] >= Nt[le] - Nkt[k, le + self.floorTb] - self.M * (1 - c[k, le + self.floorTb]))

                for le in range(len(self.Tl)):
                    for k in range(len(self.classes)):
                        mdl.add_constraint(
                            L[le] <= Nt[le] - Nkt[k, le + self.floorTb] + self.M * c[k, le + self.floorTb])

                for le in range(len(self.Tl)):
                    for k in range(len(self.classes)):
                        mdl.add_constraint(
                            Nkt[k, le + self.floorTb] == 0.5 * mdl.sum(
                                (1 + Y[i, k]) * z[i, le + self.floorTb] for i in range(len(points))))

                for le in range(len(self.Tl)):
                    mdl.add_constraint(Nt[le] == mdl.sum(z[p, le + self.floorTb] for p in range(len(points))))

                for le in range(len(self.Tl)):
                    mdl.add_constraint(l[le] == mdl.sum(c[k, le + self.floorTb] for k in range(len(self.classes))))

                for p in range(len(points)):
                    p1 = points[p]

                    for le in range(len(self.Tl)):
                        for n in self.Ar[le + self.floorTb]:
                            mdl.add_constraint(
                                np.dot(dataframe.loc[p1], a[n]) >= b[n] - self.M2 * (1 - z[p, le + self.floorTb]))

                for p in range(len(points)):
                    p1 = points[p]

                    for le in range(len(self.Tl)):
                        for m in self.Al[le + self.floorTb]:
                            mdl.add_constraint(
                                np.dot(dataframe.loc[p1] + self.eps - err * np.ones(len(self.features)), a[m]) + err <=
                                b[
                                    m] + self.M1 * (1 - z[p, le + self.floorTb]))

                for p in range(len(points)):
                    mdl.add_constraint(mdl.sum(z[p, le + self.floorTb] for le in range(len(self.Tl))) == 1)  #

                for le in range(len(self.Tl)):
                    for p in range(len(points)):
                        mdl.add_constraint(z[p, le + self.floorTb] <= l[le])

                for le in range(len(self.Tl)):
                    mdl.add_constraint(
                        l[le] * self.Nmin <= mdl.sum(z[p, le + self.floorTb] for p in range(len(points))))

                for t in self.Tb:
                    mdl.add_constraint(d[t] == mdl.sum(a[t][f] for f in self.features))

                for t in self.Tb:
                    mdl.add_constraint(b[t] <= d[t])
                for t in np.delete(self.Tb, 0):
                    mdl.add_constraint(d[t] <= d[self.parent[t]])

                randa = [np.random.uniform(1 - 1e-1, 1 + 1e-1) for t in self.Tb]
                randL = [np.random.uniform(1 - 1e-1, 1 + 1e-1) for t in self.Tl]

                for i in range(0, len(self.Tl), 2):
                    mdl.add_constraint(l[i] <= d[self.find_pt()[i + self.depth]])
                if len(self.classes) <= len(self.Tl) and self.dataset != 'seismic-bumps.csv':
                    for k in range(len(self.classes)):
                        mdl.add_constraint(1 <= mdl.sum(c[k, le + self.floorTb] for le in range(len(self.Tl))))

            # OBJECTIVE FUNCTION
            if self.version == 'multivariate':
                if self.name == 'OCT-H':
                    mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * (
                        mdl.sum(s[t][f] for t in self.Tb for f in self.features)))
                if self.name == 'S1' or self.name == 'St':
                    mdl.minimize(
                        mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * mdl.sum(d[t] for t in self.Tb))
                if self.name == 'LDA':
                    mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * (
                            mdl.sum(d[t] for t in self.Tb) + mdl.sum(
                        a_hat[t][f] for t in self.Tb for f in self.features) / len(
                        self.features * len(self.Tb))))
            elif self.version == 'univariate':
                mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * mdl.sum(d[t] for t in self.Tb))

            mdl.print_information()

        elif self.relax == 1:
            if self.version == 'univariate':
                self.eps = self.find_eps(dataframe2)
                self.M = self.find_M(dataframe2)
                self.M1 = self.find_M1(dataframe2)
                err = min(self.eps)

            elif self.version == 'multivariate':
                self.M = len(dataframe)

            mdl = Model(name=self.name)
            mdl.clear()
            # if self.version == 'multivariate':
            #    mdl.parameters.emphasis.mip = 0  # TODO SCEGLIERE EMPHASIS (MULTIVARIATE MEGLIO 4)

            Y = np.arange(len(self.classes) * len(points)).reshape(len(points), len(self.classes))
            for i in range(0, len(points)):
                for k in range(0, len(self.classes)):
                    k1 = self.classes[k]
                    if y.values[i] == k1:
                        Y[i, k] = 1
                    else:
                        Y[i, k] = -1
            # VARIABLES

            if self.version == 'univariate':
                a = []
                for t in self.Tb:
                    a.append(mdl.continuous_var_list(len(self.features), lb=0, ub=1, name='a%d' % t))
                b = mdl.continuous_var_list(self.Tb, lb=0,
                                            name='b')

            elif self.version == 'multivariate':
                a = []
                for t in self.Tb:
                    a.append(mdl.continuous_var_list(len(self.features), ub=1, lb=-1, name='a%d' % t))
                a_hat = []
                for t in self.Tb:
                    a_hat.append(mdl.continuous_var_list(len(self.features), ub=1, lb=-1,
                                                         name='a_hat%d' % t))
                if self.name == 'OCT-H' or self.name == 'S1' or self.name == 'St':
                    s = []
                    for t in self.Tb:
                        s.append(mdl.continuous_var_list(len(self.features), lb=0, ub=1, name='s%d' % t))
                b = mdl.continuous_var_list(self.Tb,
                                            name='b')

            d = mdl.continuous_var_list(self.Tb, lb=0, ub=1, name='d')

            z = mdl.continuous_var_matrix(Num_points, self.Tl, lb=0, ub=1, name='z')

            l = mdl.continuous_var_list(self.Tl, lb=0, ub=1, name='l')

            c = mdl.continuous_var_matrix(len(self.classes), self.Tl, lb=0, ub=1, name='c')

            L = mdl.continuous_var_list(self.Tl, lb=0, name='L')

            Nt = mdl.continuous_var_list(self.Tl, name='Nt')

            Nkt = mdl.continuous_var_matrix(len(self.classes), self.Tl, name='Nkt')

            # CONSTRAINTS
            if self.version == 'multivariate':

                for le in range(len(self.Tl)):
                    for k in range(len(self.classes)):
                        mdl.add_constraint(
                            L[le] >= Nt[le] - Nkt[k, le + self.floorTb] - self.M * (1 - c[k, le + self.floorTb]))

                for le in range(len(self.Tl)):
                    for k in range(len(self.classes)):
                        mdl.add_constraint(
                            L[le] <= Nt[le] - Nkt[k, le + self.floorTb] + self.M * c[k, le + self.floorTb])

                for le in range(len(self.Tl)):
                    for k in range(len(self.classes)):
                        mdl.add_constraint(
                            Nkt[k, le + self.floorTb] == 0.5 * mdl.sum(
                                (1 + Y[i, k]) * z[i, le + self.floorTb] for i in range(len(points))))

                for le in range(len(self.Tl)):
                    mdl.add_constraint(Nt[le] == mdl.sum(z[p, le + self.floorTb] for p in range(len(points))))

                for le in range(len(self.Tl)):
                    mdl.add_constraint(l[le] == mdl.sum(c[k, le + self.floorTb] for k in range(len(self.classes))))

                for p in range(len(points)):
                    mdl.add_constraint(mdl.sum(z[p, le + self.floorTb] for le in range(len(self.Tl))) == 1)

                for le in range(len(self.Tl)):
                    for p in range(len(points)):
                        mdl.add_constraint(z[p, le + self.floorTb] <= l[le])

                for le in range(len(self.Tl)):
                    mdl.add_constraint(
                        l[le] * self.Nmin <= mdl.sum(z[p, le + self.floorTb] for p in range(len(points))))

                for t in np.delete(self.Tb, 0):
                    mdl.add_constraint(d[t] <= d[self.parent[t]])

                for p in range(len(points)):
                    p1 = points[p]
                    for le in range(len(self.Tl)):
                        for n in self.Ar[le + self.floorTb]:
                            mdl.add_constraint(
                                np.dot(dataframe.loc[p1], a[n]) >= b[n] - self.bigM * (1 - z[p, le + self.floorTb]))

                for p in range(len(points)):
                    p1 = points[p]
                    for le in range(len(self.Tl)):
                        for m in self.Al[le + self.floorTb]:
                            mdl.add_constraint(
                                np.dot(dataframe.loc[p1], a[m]) + self.mu <= b[m] + (self.bigM + self.mu) * (
                                        1 - z[p, le + self.floorTb]))
                for t in self.Tb:
                    mdl.add_constraint(d[t] >= mdl.sum(a_hat[t][f] for f in self.features))

                for t in self.Tb:
                    for f in self.features:
                        mdl.add_constraint(a_hat[t][f] >= a[t][f])

                for t in self.Tb:
                    for f in self.features:
                        mdl.add_constraint(a_hat[t][f] >= -a[t][f])
                if self.name == 'OCT-H' or self.name == 'S1' or self.name == 'St':

                    for t in self.Tb:
                        for f in self.features:
                            mdl.add_constraint(-s[t][f] <= a[t][f])

                    for t in self.Tb:
                        for f in self.features:
                            mdl.add_constraint(a[t][f] <= s[t][f])

                    for t in self.Tb:
                        for f in self.features:
                            mdl.add_constraint(s[t][f] <= d[t])

                    for t in self.Tb:
                        mdl.add_constraint(mdl.sum(s[t][f] for f in self.features) >= d[t])

                for t in self.Tb:
                    mdl.add_constraint(b[t] <= d[t])

                for t in self.Tb:
                    mdl.add_constraint(b[t] >= -d[t])

                if self.name == 'St':
                    for t in self.Tb:
                        mdl.add_constraint(mdl.sum(s[t][f] for f in self.features) <= self.max_features)
                if self.name == 'S1' or self.name == 'St':
                    mdl.add_constraint(
                        mdl.sum(s[t][f] for t in self.Tb for f in self.features) <= self.max_features * len(self.Tb))

            '''elif self.version == 'univariate':
                for p in range(len(points)):
                    p1 = points[p]
                    for le in range(len(self.Tl)):
                        for n in self.Ar[le + self.floorTb]:
                            mdl.add_constraint(
                                np.dot(dataframe.loc[p1], a[n]) >= b[n] - self.M2 * (1 - z[p, le + self.floorTb]))

                for p in range(len(points)):
                    p1 = points[p]
                    for le in range(len(self.Tl)):
                        for m in self.Al[le + self.floorTb]:
                            mdl.add_constraint(
                                np.dot(dataframe.loc[p1] + self.eps - err * np.ones(len(self.features)), a[m]) + err <= b[
                                    m] + self.M1 * (1 - z[p, le + self.floorTb]))
                for t in self.Tb:
                    mdl.add_constraint(d[t] == mdl.sum(a[t][f] for f in self.features))
                for t in self.Tb:
                    mdl.add_constraint(b[t] <= d[t])

                for i in range(0, len(self.Tl), 2):
                    mdl.add_constraint(l[i] <= d[self.find_pt()[i + self.depth]])'''
            if self.version == 'univariate':
                Y = np.arange(len(self.classes) * len(points)).reshape(len(points), len(self.classes))
                for i in range(0, len(points)):
                    for k in range(0, len(self.classes)):
                        if y.values[i] == k:
                            Y[i, k] = 1
                        else:
                            Y[i, k] = -1
                for le in range(len(self.Tl)):
                    for k in range(len(self.classes)):
                        mdl.add_constraint(
                            L[le] >= Nt[le] - Nkt[k, le + self.floorTb] - self.M * (1 - c[k, le + self.floorTb]))

                for le in range(len(self.Tl)):
                    for k in range(len(self.classes)):
                        mdl.add_constraint(
                            L[le] <= Nt[le] - Nkt[k, le + self.floorTb] + self.M * c[k, le + self.floorTb])

                for le in range(len(self.Tl)):
                    for k in range(len(self.classes)):
                        mdl.add_constraint(
                            Nkt[k, le + self.floorTb] == 0.5 * mdl.sum(
                                (1 + Y[i, k]) * z[i, le + self.floorTb] for i in range(len(points))))

                for le in range(len(self.Tl)):
                    mdl.add_constraint(Nt[le] == mdl.sum(z[p, le + self.floorTb] for p in range(len(points))))

                for le in range(len(self.Tl)):
                    mdl.add_constraint(l[le] == mdl.sum(c[k, le + self.floorTb] for k in range(len(self.classes))))

                for p in range(len(points)):
                    p1 = points[p]

                    for le in range(len(self.Tl)):
                        for n in self.Ar[le + self.floorTb]:
                            mdl.add_constraint(
                                np.dot(dataframe.loc[p1], a[n]) >= b[n] - self.M2 * (1 - z[p, le + self.floorTb]))

                for p in range(len(points)):
                    p1 = points[p]

                    for le in range(len(self.Tl)):
                        for m in self.Al[le + self.floorTb]:
                            mdl.add_constraint(
                                np.dot(dataframe.loc[p1] + self.eps - err * np.ones(len(self.features)), a[m]) + err <=
                                b[
                                    m] + self.M1 * (1 - z[p, le + self.floorTb]))

                for p in range(len(points)):
                    mdl.add_constraint(mdl.sum(z[p, le + self.floorTb] for le in range(len(self.Tl))) == 1)  #

                for le in range(len(self.Tl)):
                    for p in range(len(points)):
                        mdl.add_constraint(z[p, le + self.floorTb] <= l[le])

                for le in range(len(self.Tl)):
                    mdl.add_constraint(
                        l[le] * self.Nmin <= mdl.sum(z[p, le + self.floorTb] for p in range(len(points))))

                for t in self.Tb:
                    mdl.add_constraint(d[t] == mdl.sum(a[t][f] for f in self.features))

                for t in self.Tb:
                    mdl.add_constraint(b[t] <= d[t])
                for t in np.delete(self.Tb, 0):
                    mdl.add_constraint(d[t] <= d[self.parent[t]])

                randa = [np.random.uniform(1 - 1e-1, 1 + 1e-1) for t in self.Tb]
                randL = [np.random.uniform(1 - 1e-1, 1 + 1e-1) for t in self.Tl]

                for i in range(0, len(self.Tl), 2):
                    mdl.add_constraint(l[i] <= d[self.find_pt()[i + self.depth]])

            # OBJECTIVE FUNCTION
            if self.version == 'multivariate':
                if self.name == 'OCT-H':
                    mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * (
                        mdl.sum(s[t][f] for t in self.Tb for f in self.features)))
                if self.name == 'S1' or self.name == 'St':
                    mdl.minimize(
                        mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * mdl.sum(d[t] for t in self.Tb))
                if self.name == 'LDA':
                    mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * (
                            mdl.sum(d[t] for t in self.Tb) + mdl.sum(
                        a_hat[t][f] for t in self.Tb for f in self.features) / len(
                        self.features * len(self.Tb))))
            elif self.version == 'univariate':
                mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * mdl.sum(d[t] for t in self.Tb))

            mdl.print_information()

        return mdl

    def find_cart(self, dataframe, dataframe2, y):

        mdl = self.model(dataframe, dataframe2, y)

        clf = DecisionTreeClassifier(max_depth=self.depth, min_samples_leaf=self.Nmin, random_state=1)
        clf.fit(dataframe, y)

        '''dot_data = tree.export_graphviz(clf, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render(filename="prova", directory='/Users/giuliaciarimboli/Desktop/laurea magistrale/classification trees/graphs', view=True)'''

        sk_features = clf.tree_.feature
        sk_b = clf.tree_.threshold
        sk_val = clf.tree_.value
        sk_z = clf.apply(df)
        nodes = np.append(self.Tb, self.Tl)
        idx = [0]
        j = 1
        left = clf.tree_.children_left
        right = clf.tree_.children_right
        for i in range(len(clf.tree_.children_left)):
            if idx[i] >= 0:
                node = idx[i]
            if clf.tree_.children_left[node] > 0:
                idx.insert(j, clf.tree_.children_left[node])
                j += 1
            if clf.tree_.children_right[node] > 0:
                idx.insert(j, clf.tree_.children_right[node])
                j += 1

        m = SolveSolution(mdl)
        count = 0
        j = -1
        for node in range(len(sk_features)):
            j += 1
            if sk_features[j] >= 0:
                i = list(idx).index(
                    j)  # prendo l'indice j-esimo della lista dei nodi di sklearn, equivalente al nodo oct
                feat = sk_features[j]  # è la feature da prendere nell'i esimo nodo

                m.add_var_value('a%d_%d' % (i, feat), 1)
                if self.version == 'multivariate':
                    m.add_var_value('a_hat%d_%d' % (i, feat), 1)
                    # m.add_var_value('s%d_%d' % (i, feat), 1)
                m.add_var_value(('b_%d' % i), sk_b[j])
                count += 1

        for t in self.Tb:  # len(skval)
            if t < len(sk_features):
                if sk_features[t] >= 0:
                    i = list(idx).index(t)
                    m.add_var_value(('d_%d' % i), 1)
        for leaf in self.Tl:
            m.add_var_value(('l_%d' % leaf), 1)

        jj = -1
        for node in idx:
            jj += 1
            k = np.argmax(sk_val[jj][0])
            num = np.sum(sk_val[jj][0])
            ii = list(idx).index(jj)
            if ii in self.Tl:
                m.add_var_value('c_%d_%d' % (k, ii), 1)
                m.add_var_value('Nt_%d' % ii, num)
                for kl in range(len(self.classes)):
                    m.add_var_value('Nkt_%d_%d' % (kl, ii), sk_val[jj][0][kl])
        missing = len(np.append(self.Tb, self.Tl)) - len(idx)
        for data in range(len(dataframe)):
            foglia = list(idx).index(sk_z[data]) + missing
            m.add_var_value('z_%d_%d' % (data, foglia), 1)
        mdl.add_mip_start(m)

        return mdl

    def fit_with_cart(self, dataframe, dataframe2, y):

        if self.dataset == 'wall-robot2.csv' or self.dataset == 'thyroid-new.csv':
            sol = self.model(dataframe, dataframe2, y)
        else:
            sol = self.find_cart(dataframe, dataframe2, y)
        if self.version == 'univariate':
            sol.set_time_limit(
                3600)  # TODO METTERE TIME LIMIT PER LA SOLUZIONE FINALE DEL MODELLO UNIVARIATE COL CART (1800)
            sol.export('/Users/giuliaciarimboli/Desktop')

            self.csvrow[0] = self.dataset
            self.csvrow[2] = len(self.features)
            self.csvrow[3] = self.version
            self.csvrow[4] = self.mipstart
            self.csvrow[5] = self.depth
            self.csvrow[6] = len(self.Tb)
            self.csvrow[7] = len(self.Tl)
            self.csvrow[10] = sol.number_of_variables
            self.csvrow[11] = sol.number_of_binary_variables
            self.csvrow[12] = sol.number_of_continuous_variables
            self.csvrow[13] = sol.number_of_constraints

            s = sol.solve(log_output=True)
            sol.print_solution()
            activenodes = 0
            activeleaves = 0
            for t in range(len(self.Tb)):
                if s.get_value('d_%d' % t) == 1:
                    activenodes += 1
            for leaf in self.Tl:
                if s.get_value('l_%d' % leaf) == 1:
                    activeleaves += 1
            self.csvrow[8] = activenodes
            self.csvrow[9] = activeleaves
            self.csvrow[24] = int(s.solve_details.time)
            self.csvrow[25] = s.solve_details.mip_relative_gap
            self.csvrow[26] = '?'
            self.csvrow[27] = s.objective_value

            self.draw_graph_univariate(s)
            train_error = 0
            for leaf in self.Tl:
                train_error += s.get_value('L_' + str(leaf))
            train_error = train_error / self.M
            a_test = [] * len(self.features)
            b_test = []
            c_test = []
            for t in self.Tb:
                a_list = []
                b_test.insert(t, s.get_value('b_%d' % t))
                for f in self.features:
                    a_list.insert(f, s.get_value('a%d_%d' % (t, f)))
                a_test.append(a_list)
            for leaf in self.Tl:
                c_list = []
                for k in range(len(self.classes)):
                    c_list.insert(leaf, s.get_value('c_%d_%d' % (k, leaf)))
                c_test.append(c_list)

            return a_test, b_test, c_test, train_error

        elif self.version == 'multivariate':
            sol.set_time_limit(
                300)  # TODO METTERE TIME LIMIT PER LA RISOLUZIONE DEL WARM START DEL MULTIVARIATO CON PROFONDITA 1 (240)
            sol.solve(log_output=True)
            return sol

    def find_oct_warmstart(self, dataframe, dataframe2, y):
        mod = self.find_cart(dataframe, dataframe2, y)
        mod.set_time_limit(1800)  # TODO TIME LIMIT PER RISOLVERE IL WARM START OCT CON DEPTH= D-1
        s = mod.solve(log_output=True)
        mod.print_solution()
        self.draw_graph_univariate(s)

        a_oct = [] * len(self.features)
        b_oct = []
        d_oct = []
        l_oct = []
        c_oct = [] * len(self.classes)
        z_oct = []
        Nt_oct = []
        Nkt_oct = []

        for t in self.Tb:  # inserisce valori noti
            a_list = []
            b_oct.insert(t, s.get_value('b_%d' % (t)))
            d_oct.insert(t, s.get_value('d_%d' % (t)))
            for f in self.features:
                a_list.insert(f, s.get_value('a%d_%d' % (t, f)))
            a_oct.append(a_list)

        for node in range(0, pow(2, self.depth)):  # inserisce valori in più
            a_oct.append([0] * len(self.features))
            b_oct.insert(self.Tl[-1] + node, 0)
            d_oct.insert(self.Tl[-1] + node, 0)
        j = 0
        for node in self.Tl:  # inserisce valori noti e non
            l_oct.insert(j, 0)
            l_oct.insert(j + 1, s.get_value('l_%d' % (node)))
            j += 2
        for node in self.Tl:
            c_list1 = []
            c_list2 = []
            for k in range(len(self.classes)):
                c_list1.insert(k, s.get_value('c_%d_%d' % (k, node)))
                c_list2.insert(k, 0)
            c_oct.append(c_list2)
            c_oct.append(c_list1)

        for point in range(len(dataframe)):
            for leaf in self.Tl:
                if s.get_value('z_%d_%d' % (point, leaf)) == 1:
                    z_oct.insert(point, leaf)
        i = 0
        for leaf in self.Tl:
            Nt_oct.insert(i, 0)
            i += 1
            Nt_oct.insert(i, s.get_value('Nt_%d' % (leaf)))
            i += 1
            Nkt_list = []
            for k in range(len(self.classes)):
                Nkt_list.insert(k, s.get_value('Nkt_%d_%d' % (k, leaf)))
            Nkt_oct.append([0] * len(self.classes))
            Nkt_oct.append(Nkt_list)
        return a_oct, b_oct, d_oct, l_oct, c_oct, z_oct, Nt_oct, Nkt_oct

    def fit_with_oct_mip_start(self, dataframe, dataframe2, y, warm_start):

        sol = self.model(dataframe, dataframe2, y)

        self.csvrow[0] = self.dataset
        self.csvrow[2] = len(self.features)
        self.csvrow[3] = self.version
        self.csvrow[4] = self.mipstart
        self.csvrow[5] = self.depth
        self.csvrow[6] = len(self.Tb)
        self.csvrow[7] = len(self.Tl)
        self.csvrow[10] = sol.number_of_variables
        self.csvrow[11] = sol.number_of_binary_variables
        self.csvrow[12] = sol.number_of_continuous_variables
        self.csvrow[13] = sol.number_of_constraints

        s = SolveSolution(sol)
        self.draw_graph_univariate(s)
        i = 0
        for t in self.Tb:
            s.add_var_value('b_%d' % (t), warm_start[1][t])
            s.add_var_value('d_%d' % (t), warm_start[2][t])
            for f in self.features:
                s.add_var_value('a%d_%d' % (t, f), warm_start[0][t][f])
        for leaf in self.Tl:
            s.add_var_value(('l_%d' % (leaf)), warm_start[3][i])
            i += 1
        l = 0  # indice
        for leaf in self.Tl:
            for k in range(len(self.classes)):
                s.add_var_value('c_%d_%d' % (k, leaf), warm_start[4][l][k])
            l += 1
        for point in range(len(dataframe)):
            ex_leaf = warm_start[5][point]
            son_right = 2 * ex_leaf + 2
            s.add_var_value('z_%d_%d' % (point, son_right), 1)
        i = 0
        j = 0
        for leaf in self.Tl:
            s.add_var_value('Nt_%d' % (leaf), warm_start[6][i])
            i += 1
            for k in range(len(self.classes)):
                s.add_var_value('Nkt_%d_%d' % (k, leaf), warm_start[7][j][k])

            j += 1
        print(s.check_as_mip_start())
        sol.add_mip_start(s)
        sol.set_time_limit(3600)  # TODO TIME LIMIT PER TROVARE SOLUZIONE FINALE UNIVARIATE_OCT
        # mdl.parameters.mip.tolerances.mipgap(0.1)
        # sol.parameters.emphasis.mip = 0
        print('finding solution with OCT as MIP START:')
        s = sol.solve(log_output=True)
        sol.print_solution()
        activenodes = 0
        activeleaves = 0
        for t in range(len(self.Tb)):
            if s.get_value('d_%d' % t) == 1:
                activenodes += 1
        for leaf in self.Tl:
            if s.get_value('l_%d' % leaf) == 1:
                activeleaves += 1
        self.csvrow[8] = activenodes
        self.csvrow[9] = activeleaves
        self.csvrow[24] = int(s.solve_details.time)
        self.csvrow[25] = s.solve_details.mip_relative_gap
        self.csvrow[26] = '?'
        self.csvrow[27] = s.objective_value

        # GRAPH
        self.draw_graph_univariate(s)

        train_error = 0
        for leaf in self.Tl:
            train_error += s.get_value('L_' + str(leaf))
        train_error = train_error / self.M
        a_test = [] * len(self.features)
        b_test = []
        c_test = []
        for t in self.Tb:
            a_list = []
            b_test.insert(t, s.get_value('b_%d' % t))
            for f in self.features:
                a_list.insert(f, s.get_value('a%d_%d' % (t, f)))
            a_test.append(a_list)
        for leaf in self.Tl:
            c_list = []
            for k in range(len(self.classes)):
                c_list.insert(leaf, s.get_value('c_%d_%d' % (k, leaf)))
            c_test.append(c_list)

        return a_test, b_test, c_test, train_error

    def draw_graph_univariate(self, sol):
        g = pgv.AGraph(directed=True)  # initialize the graph

        nodes = np.append(self.Tb, self.Tl)
        for n in nodes:  # the graph has a node for eache node of the tree
            g.add_node(n, shape='circle', size=24)
            if n != 0:
                father = ceil(n / 2) - 1
                g.add_edge(father, n)
        for t in self.Tb:
            # if mdl.solution.get_value('d_' + str(t))==0:
            # g.get_node(t).attr['color']='red'
            for f in range(len(self.features)):
                if sol.get_value('a' + str(t) + '_' + str(f)) == 1:
                    g.get_node(t).attr['label'] = str('X[%d]' % (f)) + str(' < ') + str(
                        '%.3f' % (sol.get_value('b_' + str(t))))
        for leaf in self.Tl:
            if sol.get_value('l_' + str(leaf)) == 0:  # these leaves haven't got points
                g.get_node(leaf).attr['color'] = 'red'
        for leaf in self.Tl:
            s = []
            for k in range(len(self.classes)):
                s.append(round(sol.get_value('Nkt_' + str(k) + '_' + str(leaf))))
            for k in range(len(self.classes)):
                if sol.get_value('c_' + str(k) + '_' + str(leaf)) == 1:
                    g.get_node(leaf).attr['label'] = str(s) + '\\n' + 'class %d' % (k)
        g.layout(prog='dot')
        g.draw('/Users/giuliaciarimboli/Desktop/%s_%s_%d.pdf' % (self.dataset, self.mipstart, self.depth))

        return g

    def fit_multivariate(self, dataframe, dataframe2, y, d, modello):

        ordine_l = [0, 1, 4, 3, 10, 9, 8, 6, 22, 21, 20, 19, 18, 17, 16, 15]
        ordine_r = [0, 2, 6, 5, 14, 13, 12, 11, 20, 29, 28, 27, 26, 25, 24, 23]

        mm = SolveSolution(modello)

        T = pow(2, (d + 1)) - 1  # nodes number
        floorTb = int(floor(T / 2))  # number of branch nodes
        Tb = np.arange(0, floorTb)  # range branch nodes
        Tl = np.arange(floorTb, T)  # range leaf nodes
        classes = np.unique(y.values)  # possible labels of classification

        lista_df = []
        lista_y = []
        y = pd.DataFrame(y)
        df_ = []
        y_ = []
        i = 0
        for t in y.index:
            df_.insert(-1, dataframe.loc[i])
            i += 1
            y_.insert(-1, y.loc[t])
        df_ = pd.DataFrame(df_)
        y_ = pd.DataFrame(y_)
        lista_df.insert(0, df_)
        lista_y.insert(0, y_)
        for t in range(int((len(Tb) - 1) / 2) + 1):
            yy = lista_y[t]
            df_split1 = []
            df_split2 = []
            y_1 = []
            y_2 = []
            ind = lista_y[t].index
            ind_df = lista_df[t].index
            if len(lista_y[t]) >= self.Nmin:
                '''for f in self.features:
                    mm.add_var_value('a%d_%d' % (ordine_l[t], f), 0)
                    mm.add_var_value('a_hat%d_%d' % (ordine_l[t], f), 0)
                mm.add_var_value('b_%d' % (ordine_l[t]), 0)
                mm.add_var_value('d_%d' % (ordine_l[t]), 0)
                if 2 * ordine_l[t] + 1 in Tl:
                    leaf = 2 * ordine_l[t] + 1
                    for l in range(leaf, leaf+2):
                        mm.add_var_value('Nt_%d' % l, 0)
                        mm.add_var_value('l_%d' % l, 0)
                        for k in range(len(classes)):
                            mm.add_var_value('c_%d_%d' % (k, l),
                                             0)
                            mm.add_var_value('Nkt_%d_%d' % (k, l),
                                             0)


                        for n in range(0, len(dataframe)):
                            mm.add_var_value('z_%d_%d' % (n, l), 0)
                            #mm.add_var_value('z_%d_%d'%(n, (leaf+1)), 0)
                lista_df.insert(1, [])
                lista_df.insert(2, [])
                lista_y.insert(1, [])
                lista_y.insert(2,[])
            else:'''
                mdl = self.fit_with_cart(lista_df[t], lista_df[t], lista_y[t])
                print('risolvo')
                cl = yy[0].unique()
                cl.sort()
                for f in self.features:
                    mm.add_var_value('a%d_%d' % (ordine_l[t], f), mdl.solution.get_value('a0_%d' % f))
                    mm.add_var_value('a_hat%d_%d' % (ordine_l[t], f), mdl.solution.get_value('a_hat0_%d' % f))
                    if self.name == 'OCT-H' or self.name == 'S1' or self.name == 'St':
                        mm.add_var_value('s%d_%d' % (ordine_l[t], f), mdl.solution.get_value('s0_%d' % f))
                mm.add_var_value('b_%d' % (ordine_l[t]), mdl.solution.get_value('b_0'))
                mm.add_var_value('d_%d' % (ordine_l[t]), mdl.solution.get_value('d_0'))
                if 2 * ordine_l[t] + 1 in Tl:
                    leaf = 2 * ordine_l[t] + 1
                    mm.add_var_value('Nt_%d' % leaf, mdl.solution.get_value('Nt_1'))
                    mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_1'))

                    for k in range(len(cl)):
                        mm.add_var_value('c_%d_%d' % (list(classes).index(cl[k]), leaf),
                                         mdl.solution.get_value('c_%d_1' % k))
                        mm.add_var_value('Nkt_%d_%d' % (list(classes).index(cl[k]), leaf),
                                         mdl.solution.get_value('Nkt_%d_1' % k))
                    kl = list(set(classes) - set(cl))
                    for k2 in range(len(kl)):
                        mm.add_var_value('c_%d_%d' % (list(classes).index(kl[k2]), leaf), 0)
                        mm.add_var_value('Nkt_%d_%d' % (list(classes).index(kl[k2]), leaf), 0)
                    for n in range(len(lista_df[t])):
                        mm.add_var_value('z_%d_%d' % (ind_df[n], leaf), mdl.solution.get_value('z_%d_1' % n))
                    ind_miss = list(set(list(np.array(np.arange(0, len(dataframe))))) - set(ind_df))
                    for n in ind_miss:
                        mm.add_var_value('z_%d_%d' % (n, leaf), 0)

                if 2 * ordine_l[t] + 2 in Tl:
                    leaf = 2 * ordine_l[t] + 2
                    mm.add_var_value('Nt_%d' % leaf, mdl.solution.get_value('Nt_2'))
                    mm.add_var_value('l_%d' % leaf, mdl.solution.get_value('l_2'))
                    for k in range(len(cl)):
                        mm.add_var_value('c_%d_%d' % (list(classes).index(cl[k]), leaf),
                                         mdl.solution.get_value('c_%d_2' % k))
                        mm.add_var_value('Nkt_%d_%d' % (list(classes).index(cl[k]), leaf),
                                         mdl.solution.get_value('Nkt_%d_2' % k))
                    kl = list(set(classes) - set(cl))
                    for k2 in range(len(kl)):
                        mm.add_var_value('c_%d_%d' % (list(classes).index(kl[k2]), leaf), 0)
                        mm.add_var_value('Nkt_%d_%d' % (list(classes).index(kl[k2]), leaf), 0)
                    for n in range(len(lista_df[t])):
                        mm.add_var_value('z_%d_%d' % (ind_df[n], leaf), mdl.solution.get_value('z_%d_2' % (n)))
                    ind_miss = list(set(list(np.array(np.arange(0, len(dataframe))))) - set(ind_df))
                    for n in ind_miss:
                        mm.add_var_value('z_%d_%d' % (n, leaf), 0)
                for i in range(len(lista_df[t])):
                    j = ind[i]
                    m = ind_df[i]
                    if mdl.solution.get_value('z_%d_1' % i) == 1:
                        df_split1.insert(-1, lista_df[t].loc[m])
                        y_1.insert(-1, lista_y[t].loc[j])
                    else:
                        df_split2.insert(-1, lista_df[t].loc[m])
                        y_2.insert(-1, lista_y[t].loc[j])
                df_1 = pd.DataFrame(df_split1)
                df_2 = pd.DataFrame(df_split2)
                y_1 = pd.DataFrame(y_1)
                y_2 = pd.DataFrame(y_2)
                lista_df.insert(1, df_1)
                lista_df.insert(2, df_2)
                lista_y.insert(1, y_1)
                lista_y.insert(2, y_2)
            else:
                lista_df.insert(1, [])
                lista_df.insert(2, [])
                lista_y.insert(1, [])
                lista_y.insert(2, [])
                for f in self.features:
                    mm.add_var_value('a%d_%d' % (ordine_l[t], f), 0)
                    mm.add_var_value('a_hat%d_%d' % (ordine_l[t], f), 0)
                mm.add_var_value('b_%d' % (ordine_l[t]), 0)
                mm.add_var_value('d_%d' % (ordine_l[t]), 0)

        lista_df_r = []
        lista_y_r = []
        lista_df_r.insert(0, lista_df[0])
        lista_df_r.insert(1, lista_df[-1])
        lista_y_r.insert(0, lista_y[0])
        lista_y_r.insert(1, lista_y[-1])

        for t in range(1, int((len(Tb) - 1) / 2) + 1):
            yy = lista_y_r[t]

            df_split1 = []
            df_split2 = []
            y_1 = []
            y_2 = []
            ind = lista_y_r[t].index
            ind_df = lista_df_r[t].index
            if len(lista_y_r[t]) >= self.Nmin:

                '''for f in self.features:
                    mm.add_var_value('a%d_%d' % (ordine_l[t], f), 0)
                    mm.add_var_value('a_hat%d_%d' % (ordine_l[t], f), 0)
                mm.add_var_value('b_%d' % (ordine_l[t]), 0)
                mm.add_var_value('d_%d' % (ordine_l[t]), 0)
                if 2 * ordine_l[t] + 1 in Tl:
                    leaf = 2 * ordine_l[t] + 1
                    for l in range(leaf, leaf + 2):
                        mm.add_var_value('Nt_%d' % l, 0)
                        mm.add_var_value('l_%d' % l, 0)
                        for k in range(len(classes)):
                            mm.add_var_value('c_%d_%d' % (k, l),
                                             0)
                            mm.add_var_value('Nkt_%d_%d' % (k, l),
                                             0)
                        for n in range(0, len(dataframe)):
                            mm.add_var_value('z_%d_%d' % (n, l), 0)
                lista_df_r.insert(1, [])
                lista_df_r.insert(2, [])
                lista_y_r.insert(1, [])
                lista_y_r.insert(2, [])
            else:'''
                print('risolvo')
                mdl = self.fit_with_cart(lista_df_r[t], lista_df_r[t], lista_y_r[t])
                cl = yy[0].unique()
                cl.sort()

                for f in self.features:
                    mm.add_var_value('a%d_%d' % (ordine_r[t], f), mdl.solution.get_value('a0_%d' % f))
                    mm.add_var_value('a_hat%d_%d' % (ordine_r[t], f), mdl.solution.get_value('a_hat0_%d' % f))
                    if self.name == 'OCT-H' or self.name == 'S1' or self.name == 'St':
                        mm.add_var_value('s%d_%d' % (ordine_r[t], f), mdl.solution.get_value('s0_%d' % f))
                mm.add_var_value('b_%d' % (ordine_r[t]), mdl.solution.get_value('b_0'))
                mm.add_var_value('d_%d' % (ordine_r[t]), mdl.solution.get_value('d_0'))
                if 2 * ordine_r[t] + 1 in Tl:
                    leaf = 2 * ordine_r[t] + 1
                    mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_1'))
                    mm.add_var_value('Nt_%d' % (leaf), mdl.solution.get_value('Nt_1'))

                    for k in range(len(cl)):
                        mm.add_var_value('c_%d_%d' % (list(classes).index(cl[k]), leaf),
                                         mdl.solution.get_value('c_%d_1' % k))
                        mm.add_var_value('Nkt_%d_%d' % (list(classes).index(cl[k]), leaf),
                                         mdl.solution.get_value('Nkt_%d_1' % k))
                    kl = list(set(classes) - set(cl))
                    for k2 in range(len(kl)):
                        mm.add_var_value('c_%d_%d' % (list(classes).index(kl[k2]), leaf), 0)
                        mm.add_var_value('Nkt_%d_%d' % (list(classes).index(kl[k2]), leaf), 0)
                    mm.add_var_value('l_%d' % leaf, mdl.solution.get_value('l_1'))
                    for n in range(len(lista_df_r[t])):
                        mm.add_var_value('z_%d_%d' % (ind_df[n], leaf), mdl.solution.get_value('z_%d_1' % n))
                    ind_miss = list(set(list(np.array(np.arange(0, len(dataframe))))) - set(ind_df))
                    for n in ind_miss:
                        mm.add_var_value('z_%d_%d' % (n, leaf), 0)
                if 2 * ordine_r[t] + 2 in Tl:
                    leaf = 2 * ordine_r[t] + 2
                    mm.add_var_value('l_%d' % leaf, mdl.solution.get_value('l_2'))
                    mm.add_var_value('Nt_%d' % leaf, mdl.solution.get_value('Nt_2'))
                    for k in range(len(cl)):
                        mm.add_var_value('c_%d_%d' % (list(classes).index(cl[k]), leaf),
                                         mdl.solution.get_value('c_%d_2' % k))
                        mm.add_var_value('Nkt_%d_%d' % (list(classes).index(cl[k]), leaf),
                                         mdl.solution.get_value('Nkt_%d_2' % k))
                    kl = list(set(classes) - set(cl))
                    for k2 in range(len(kl)):
                        mm.add_var_value('c_%d_%d' % (list(classes).index(kl[k2]), leaf), 0)
                        mm.add_var_value('Nkt_%d_%d' % (list(classes).index(kl[k2]), leaf), 0)
                    for n in range(len(lista_df_r[t])):
                        mm.add_var_value('z_%d_%d' % (ind_df[n], leaf), mdl.solution.get_value('z_%d_2' % n))
                    ind_miss = list(set(list(np.array(np.arange(0, len(dataframe))))) - set(ind_df))

                    for n in ind_miss:
                        mm.add_var_value('z_%d_%d' % (n, leaf), 0)

                for i in range(len(lista_df_r[t])):
                    j = ind[i]
                    m = ind_df[i]
                    if mdl.solution.get_value('z_%d_1' % i) == 1:
                        df_split1.insert(-1, lista_df_r[t].loc[m])
                        y_1.insert(-1, lista_y_r[t].loc[j])
                    else:
                        df_split2.insert(-1, lista_df_r[t].loc[m])
                        y_2.insert(-1, lista_y_r[t].loc[j])
                df_1 = pd.DataFrame(df_split1)
                df_2 = pd.DataFrame(df_split2)
                y_1 = pd.DataFrame(y_1)
                y_2 = pd.DataFrame(y_2)
                lista_df_r.insert(1, df_1)
                lista_df_r.insert(2, df_2)
                lista_y_r.insert(1, y_1)
                lista_y_r.insert(2, y_2)
            else:
                lista_df_r.insert(1, [])
                lista_df_r.insert(2, [])
                lista_y_r.insert(1, [])
                lista_y_r.insert(2, [])
                for f in self.features:
                    mm.add_var_value('a%d_%d' % (ordine_r[t], f), 0)
                    mm.add_var_value('a_hat%d_%d' % (ordine_r[t], f), 0)
                mm.add_var_value('b_%d' % (ordine_r[t]), 0)
                mm.add_var_value('d_%d' % (ordine_r[t]), 0)

            # GRAPH WARM START
            g = pgv.AGraph(directed=True)  # initialize the graph
            nodes = np.append(Tb, Tl)
            for n in nodes:  # the graph has a node for each node of the tree
                g.add_node(n, shape='circle', size=8)

                if n != 0:
                    father = ceil(n / 2) - 1
                    g.add_edge(father, n)

            for t in Tb:
                coeff = []
                feat = []
                # if mdl.solution.get_value('d_' + str(t))==0:
                # g.get_node(t).attr['color']='red'
                for f in range(len(self.features)):
                    if mm.get_value('a' + str(t) + '_' + str(f)) != 0:
                        coeff.insert(-1, '%.3f' % (mm.get_value('a' + str(t) + '_' + str(f))))
                        feat.insert(-1, f)
                g.get_node(t).attr['label'] = str(coeff) + '*X' + str(feat) + str('<=') + str(
                    '%.3f' % (mm.get_value('b_' + str(t))))
            for leaf in Tl:
                if mm.get_value('l_' + str(leaf)) == 0:  # these leaves haven't got points
                    g.get_node(leaf).attr['color'] = 'red'
            for leaf in Tl:
                s = []
                for k in range(len(classes)):
                    s.append(round(mm.get_value('Nkt_' + str(k) + '_' + str(leaf))))
                for k in range(len(classes)):
                    if mm.get_value('c_' + str(k) + '_' + str(leaf)) == 1:
                        g.get_node(leaf).attr['label'] = str(s) + '\\n' + 'class %d' % (classes[k])
            g.layout(prog='dot')
            g.draw('/Users/giuliaciarimboli/Desktop/WarmStart_%s_%s_%d.pdf' % (self.dataset, self.name, d))

        print('la soluzione warm start:', mm)

        print(mm.check_as_mip_start())
        modello.add_mip_start(mm)

        modello.set_time_limit(3600)  # TODO TIME LIMIT PER TROVARE SOLUZIONE FINALE MULTIVARIATE

        # modello.parameters.emphasis.mip = 4
        self.csvrow[0] = self.dataset
        self.csvrow[2] = len(self.features)
        self.csvrow[3] = self.version
        self.csvrow[4] = self.name
        self.csvrow[5] = d
        self.csvrow[6] = len(Tb)
        self.csvrow[7] = len(Tl)
        self.csvrow[10] = modello.number_of_variables
        self.csvrow[11] = modello.number_of_binary_variables
        self.csvrow[12] = modello.number_of_continuous_variables
        self.csvrow[13] = modello.number_of_constraints

        s = modello.solve(log_output=True)

        activenodes = 0
        activeleaves = 0
        for t in range(len(Tb)):
            if s.get_value('d_%d' % t) == 1:
                activenodes += 1
        for leaf in Tl:
            if s.get_value('l_%d' % leaf) == 1:
                activeleaves += 1
        self.csvrow[8] = activenodes
        self.csvrow[9] = activeleaves
        self.csvrow[24] = int(modello.solve_details.time)
        self.csvrow[25] = modello.solve_details.mip_relative_gap
        self.csvrow[26] = '?'  # Cplex.get_dettime()
        self.csvrow[27] = modello.objective_value

        modello.print_solution()

        train_error = 0
        for leaf in Tl:
            train_error += s.get_value('L_' + str(leaf))
        train_error = train_error / len(y)
        print('train_error:', train_error)
        a_test = [] * len(self.features)
        b_test = []
        c_test = []
        for t in Tb:
            a_list = []
            b_test.insert(t, s.get_value('b_%d' % t))
            for f in self.features:
                a_list.insert(f, s.get_value('a%d_%d' % (t, f)))
            a_test.append(a_list)
        for leaf in Tl:
            c_list = []
            for k in range(len(classes)):
                c_list.insert(leaf, s.get_value('c_%d_%d' % (k, leaf)))
            c_test.append(c_list)
        # GRAPH
        g = pgv.AGraph(directed=True)  # initialize the graph

        nodes = np.append(Tb, Tl)
        for n in nodes:  # the graph has a node for eache node of the tree
            g.add_node(n, shape='circle', size=8)

            if n != 0:
                father = ceil(n / 2) - 1
                g.add_edge(father, n)

        for t in Tb:
            coeff = []
            feat = []
            # if mdl.solution.get_value('d_' + str(t))==0:
            # g.get_node(t).attr['color']='red'
            for f in range(len(self.features)):
                if modello.solution.get_value('a' + str(t) + '_' + str(f)) != 0:
                    coeff.insert(-1, '%.3f' % (modello.solution.get_value('a' + str(t) + '_' + str(f))))
                    feat.insert(-1, f)
            g.get_node(t).attr['label'] = str(coeff) + '*X' + str(feat) + str('<=') + str(
                '%.3f' % (modello.solution.get_value('b_' + str(t))))
        for leaf in Tl:
            if modello.solution.get_value('l_' + str(leaf)) == 0:  # these leaves haven't got points
                g.get_node(leaf).attr['color'] = 'red'
        for leaf in Tl:
            s = []
            for k in range(len(classes)):
                s.append(round(modello.solution.get_value('Nkt_' + str(k) + '_' + str(leaf))))
            for k in range(len(classes)):
                if modello.solution.get_value('c_' + str(k) + '_' + str(leaf)) == 1:
                    g.get_node(leaf).attr['label'] = str(s) + '\\n' + 'class %d' % (classes[k])
        g.layout(prog='dot')
        g.draw('/Users/giuliaciarimboli/Desktop/solfinale_%s_%s_%d.pdf' % (self.dataset, self.name, d))

        return a_test, b_test, c_test, train_error

    def test(self, dataframe, dataframe2, y, d, modello, dataframe_test, y_test, warm_start, relaxation_value):

        T = pow(2, (d + 1)) - 1  # nodes number
        floorTb = int(floor(T / 2))  # number of branch nodes
        Tb = np.arange(0, floorTb)  # range branch nodes
        Tl = np.arange(floorTb, T)  # range leaf nodes
        classes = np.unique(y.values)  # possible labels of classification

        if self.version == 'multivariate':
            a_test, b_test, c_test, train_error = self.fit_multivariate(dataframe, dataframe2, y, d, modello)

        elif self.version == 'univariate':
            if self.mipstart == 'CART':
                a_test, b_test, c_test, train_error = self.fit_with_cart(dataframe, dataframe2, y)
            else:
                a_test, b_test, c_test, train_error = self.fit_with_oct_mip_start(dataframe, dataframe2, y, warm_start)
        ''' leaves = []
        prediction = []
        apply = np.zeros((len(dataframe_test), d + 1), dtype=np.int8)
        for p in range(len(dataframe_test)):
            for i in range(d):
                j = int(apply[p][i])
                if np.dot(a_test[j], dataframe_test.loc[p]) + 1e-7 < b_test[j]:
                    apply[p][i + 1] = 2 * j + 1
                else:
                    apply[p][i + 1] = 2 * j + 2
            leaves.insert(p, apply[p][d])'''
        apply = self.apply(d, dataframe_test, a_test, b_test)
        leaves = self.leaves(apply, dataframe_test)
        '''count = 0
        for p in leaves:
            leaf = list(Tl).index(p)
            for k in range(len(classes)):
                if c_test[leaf][k] == 1:
                    prediction.insert(count, classes[k])
                    count += 1'''
        prediction = self.prediction(leaves, Tl, classes, c_test)

        '''print(leaves)
        print(prediction)
        print(list(y_test))
        errors = 0
        for p in range(len(dataframe_test)):
            if prediction[p] != list(y_test)[p]:
                errors += 1
        test_error = errors / len(y_test)
        print('test_error:', test_error)
        print('train_error:', train_error)'''

        test_error = self.testerror(prediction, dataframe_test, y_test)
        '''
        confusion_matrix = np.zeros((4))
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(len(prediction)):
            if prediction[i] == 1 and list(y_test)[i] == 1:
                tp += 1
            elif prediction[i] == 1 and list(y_test)[i] == 0:
                fp += 1
            elif prediction[i] == 0 and list(y_test)[i] == 1:
                fn += 1
            elif prediction[i] == 0 and list(y_test)[i] == 0:
                tn += 1
        confusion_matrix[0] = tp
        confusion_matrix[1] = fp
        confusion_matrix[2] = fn
        confusion_matrix[3] = tn

        print(confusion_matrix)'''

        tp, fp, fn, tn, confusion_matrix = self.confusionmatrix(prediction, y_test)
        print(apply)
        print(prediction)
        print(list(y_test))
        print(confusion_matrix)
        self.csvrow[1] = len(dataframe) + len(dataframe_test)
        self.csvrow[14] = train_error
        self.csvrow[19] = test_error
        self.csvrow[20] = tp
        self.csvrow[21] = fp
        self.csvrow[22] = fn
        self.csvrow[23] = tn

        apply_train = self.apply(d, dataframe, a_test, b_test)
        leaves_train = self.leaves(apply_train, dataframe)
        prediction_train = self.prediction(leaves_train, Tl, classes, c_test)
        tp_train, fp_train, fn_train, tn_train, confusion_matrix_train = self.confusionmatrix(prediction_train, y)

        self.csvrow[15] = tp_train
        self.csvrow[16] = fp_train
        self.csvrow[17] = fn_train
        self.csvrow[18] = tn_train
        self.csvrow[28] = relaxation_value
        self.csvrow[29] = self.max_features
        self.csvrow[30] = self.alpha

        print('train')
        print(apply_train)
        print(leaves_train)
        print(prediction_train)
        print(list(y))
        print(confusion_matrix_train)
        self.write_on_csv()

        return

    def apply(self, d, dataframe_test, a_test, b_test):
        leaves = []
        prediction = []
        apply = np.zeros((len(dataframe_test), d + 1), dtype=np.int8)
        for p in range(len(dataframe_test)):
            for i in range(d):
                j = int(apply[p][i])
                if np.dot(a_test[j], dataframe_test.loc[p]) + 1e-7 < b_test[j]:
                    apply[p][i + 1] = 2 * j + 1
                else:
                    apply[p][i + 1] = 2 * j + 2
            leaves.insert(p, apply[p][d])

        return apply

    def leaves(self, apply, dataframe_test):
        leaves = []
        for p in range(len(dataframe_test)):
            leaves.insert(p, apply[p][d])
        return leaves

    def prediction(self, leaves, Tl, classes, c_test):
        count = 0
        prediction = []

        for p in leaves:
            leaf = list(Tl).index(p)
            for k in range(len(classes)):
                if c_test[leaf][k] == 1:
                    prediction.insert(count, classes[k])
                    count += 1
        return prediction

    def testerror(self, prediction, dataframe_test, y_test):
        errors = 0
        for p in range(len(y_test)):
            if prediction[p] != list(y_test.loc[p])[0]:
                errors += 1
        test_error = errors / len(y_test)
        return test_error

    def confusionmatrix(self, prediction, y_test):
        confusion_matrix = np.zeros((4))
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(len(prediction)):

            if prediction[i] == 1 and list(y_test.loc[i])[0] == 1:
                print('tp')
                tp += 1
            elif prediction[i] == 1 and list(y_test.loc[i])[0] == 0:
                fp += 1
                print('fp')
            elif prediction[i] == 0 and list(y_test.loc[i])[0] == 1:
                fn += 1
                print('fn')
            elif prediction[i] == 0 and list(y_test.loc[i])[0] == 0:
                tn += 1
                print('tn')
        confusion_matrix[0] = tp
        confusion_matrix[1] = fp
        confusion_matrix[2] = fn
        confusion_matrix[3] = tn
        return tp, fp, fn, tn, confusion_matrix

    def write_on_csv(self):
        with open('OptimalTreeResults.csv', 'a',
                  newline='') as csvfile:  # '/Users/giuliaciarimboli/Desktop/OptimalTreeResults.csv'
            writer = csv.reader(csvfile, delimiter=',')
            writer = csv.writer(csvfile)
            writer.writerow(self.csvrow)
        return
    
    

def divide_train_and_test(X,y, splitPercentage, stratify = None):    
    if stratify == None:
        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=1-splitPercentage, random_state=42, stratify = stratify)
    else:
        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=1-splitPercentage, random_state=42, stratify = y)
    return xTrain, yTrain, xTest, yTest

def from_regression_to_class(yRegr, bins, labels):
    yClass = yRegr.reshape(-1,)
    y = np.asarray(pd.cut(yClass, bins = bins, labels = labels))
    y = y.reshape(-1,1)
    return y
np.random.seed(8)

df = pd.read_excel('Neurologici_Nuovi.xlsx')

y = df["COD_80"]
y = y.values.reshape(-1,1)
X = df.drop(columns = ["COD_80"])
X = X.values
threshold = 80
print(np.sum(y>=threshold))
print(np.sum(y<threshold))

bins = [0, threshold, 101]
labels = [0,1]
y = from_regression_to_class(y, bins, labels)


splitPercentage = 0.7
xTrain, yTrain, xTest, yTest = divide_train_and_test(X,y, splitPercentage, stratify = None)
xTrain=pd.DataFrame(xTrain)
xTest=pd.DataFrame(xTest)
scaler = MinMaxScaler()
df_scaled = scaler.fit(xTrain)
df = scaler.transform(xTrain)
df = pd.DataFrame(df)
df2 = scaler.transform(xTrain)
df2 = pd.DataFrame(df2)
df_test = scaler.transform(xTest)  # apply same transformation to test set
yTest=pd.DataFrame(yTest)
yTrain=pd.DataFrame(yTrain)

for i in range(len(df_test)):
    for j in range(len(df_test[0])):
        if df_test[i][j] > 1:
            df_test[i][j] = 1
        elif df_test[i][j] < 0:
            df_test[i][j] = 0
df_test = pd.DataFrame(df_test)
print(len(df))
d = 2
N = 1  # int(3 / 100 * (len(df) + len(df_test)))
F = 8  # len(df.columns)
names= ['LDA', 'S1', 'St']
a = 3
v = 'multivariate'
trainset= 'neurologici'
for n in names:
    if names != 'OCT-H' or names!= 'LDA':
        print('RISOLVO IL MODELLO %s %s PER IL DATASET %s CON PROFONDITA %d' % (v, n, trainset, d))
        t = OptimalTree(depth=d, Nmin=N, alpha=a, max_features=F, version=v, name=n, dataset=trainset)
        modello = t.model(df, df2, yTrain)
        warm = OptimalTree(depth=1, alpha=a, Nmin=N, max_features=F, name=n, dataset=trainset)
        ws = warm.test(df, df2, yTrain, d, modello, df_test, yTest, None, 0)
    else:
        t = OptimalTree(depth=d, Nmin=N, alpha=a, max_features=-1, version=v, name=n, dataset=trainset)
        modello = t.model(df, df2, yTrain)
        warm = OptimalTree(depth=1, alpha=a, Nmin=N, max_features=1, name=n, dataset=trainset)
        ws = warm.test(df, df2, yTrain, d, modello, df_test, yTest, None, 0)

