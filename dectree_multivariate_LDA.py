from math import floor
from math import ceil
from time import time

import numpy as np
import pandas as pd
from docplex.mp.model import Model
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from docplex.mp.solution import SolveSolution
import pygraphviz as pgv
from sklearn.tree import DecisionTreeClassifier



class OptimalTree(BaseEstimator):

    def __init__(self, depth=2, alpha=0.0, Nmin=1):

        self.depth = depth
        self.alpha = alpha
        self.Nmin = Nmin

        self.mu = 0.005
        self.bigM = 2

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

    def model(self, dataframe, y):

        np.random.seed(1)
        self.Tb, self.Tl, self.classes, self.features = self.find_T(dataframe, y)
        self.parent = self.find_pt()
        self.Al, self.Ar = self.find_anc()
        self.M = len(dataframe)
        points = dataframe.index
        Num_points = len(dataframe)  # number of points
        mdl = Model(name='OCT-H')
        mdl.parameters.emphasis.mip = 4
        Y = np.arange(len(self.classes) * len(points)).reshape(len(points), len(self.classes))
        for i in range(0, len(points)):
            for k in range(0, len(self.classes)):
                k1 = self.classes[k]
                if y.values[i] == k1:
                    Y[i, k] = 1
                else:
                    Y[i, k] = -1

        # VARIABLES

        a = []
        for t in self.Tb:
            a.append(mdl.continuous_var_list(len(self.features), ub=1, lb=-1, name='a%d' % t))
        a_hat = []
        for t in self.Tb:
            a_hat.append(mdl.continuous_var_list(len(self.features), ub=1, lb=-1, name='a_hat%d' % t)) 

        b = mdl.continuous_var_list(self.Tb, name='b')

        d = mdl.binary_var_list(self.Tb, name='d')

        z = mdl.binary_var_matrix(Num_points, self.Tl, name='z')

        l = mdl.binary_var_list(self.Tl, name='l')

        c = mdl.binary_var_matrix(len(self.classes), self.Tl, name='c')

        L = mdl.continuous_var_list(self.Tl, lb=0, name='L')

        Nt = mdl.continuous_var_list(self.Tl, name='Nt')

        Nkt = mdl.continuous_var_matrix(len(self.classes), self.Tl, name='Nkt')

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
                        np.dot(dataframe.loc[p1], a[n]) >= b[n] - self.bigM * (1 - z[p, le + self.floorTb]))

        for p in range(len(points)):
            p1 = points[p]
            for le in range(len(self.Tl)):
                for m in self.Al[le + self.floorTb]:
                    mdl.add_constraint(
                        np.dot(dataframe.loc[p1], a[m]) + self.mu <= b[m] + (self.bigM + self.mu) * (
                                    1 - z[p, le + self.floorTb]))

        for p in range(len(points)):
            mdl.add_constraint(mdl.sum(z[p, le + self.floorTb] for le in range(len(self.Tl))) == 1)

        for le in range(len(self.Tl)):
            for p in range(len(points)):
                mdl.add_constraint(z[p, le + self.floorTb] <= l[le])

        for le in range(len(self.Tl)):
            mdl.add_constraint(l[le] * self.Nmin <= mdl.sum(z[p, le + self.floorTb] for p in range(len(points))))

        for t in self.Tb:
            mdl.add_constraint(d[t] >= mdl.sum(a_hat[t][f] for f in self.features))

        for t in self.Tb:
            for f in self.features:
                mdl.add_constraint(a_hat[t][f] >= a[t][f])

        for t in self.Tb:
            for f in self.features:
                mdl.add_constraint(a_hat[t][f] >= -a[t][f])

        for t in self.Tb:
            mdl.add_constraint(b[t] <= d[t])

        for t in self.Tb:
            mdl.add_constraint(b[t] >= -d[t])

        for t in np.delete(self.Tb, 0):
            mdl.add_constraint(d[t] <= d[self.parent[t]])

        randa = [np.random.uniform(1 - 1e-1, 1 + 1e-1) for t in self.Tb]
        randL = [np.random.uniform(1 - 1e-1, 1 + 1e-1) for t in self.Tl]

        #for i in range(0, len(self.Tl), 2):
        #    mdl.add_constraint(l[i] <= d[self.find_pt()[i + self.depth]])
        # vincolo giorgio
        #if len(self.classes) <= len(self.Tl):
        #    for k in range(len(self.classes)):
        #        mdl.add_constraint(1 <= mdl.sum(c[k, le + self.floorTb] for le in range(len(self.Tl))))
        # mdl.add_constraint(mdl.sum(d[t] for t in self.Tb)>=1)

        #        mdl.add_constraint(mdl.sum(l[leaf] for leaf in range(len(self.Tl))) <= len(self.classes)) #vincolo prof

        # for t in self.Tb:
        #    mdl.add_constraint(mdl.sum(s[t][f] for f in self.features) <= self.max_features)
        # for f in self.features:
        #    mdl.add_constraint(mdl.sum(s[t][f] for t in self.Tb) <= 1)

        # mdl.add_constraint(mdl.sum(s[t][f] for t in self.Tb for f in self.features) <= self.max_features*len(self.Tb))

        # OBJECTIVE FUNCTION
        #mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))))
        mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * (
                    mdl.sum(d[t] for t in self.Tb) + mdl.sum(a_hat[t][f] for t in self.Tb for f in self.features) / len(
                self.features * len(self.Tb))))
        # mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * mdl.sum(d[t] for t in range(len(self.Tb))))
        # mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * ( mdl.sum(a_hat[t][f] for t in self.Tb for f in self.features)))
        # mdl.minimize(mdl.sum(L[le]*randL[le] for le in range(len(self.Tl))) + self.alpha * mdl.sum(d[t]*randa[t] for t in self.Tb))

        mdl.print_information()

        return mdl

    def find_cart(self, dataframe, y):

        mdl = self.model(dataframe, y)

        clf = DecisionTreeClassifier(max_depth=self.depth, min_samples_leaf=self.Nmin, random_state=1)
        clf.fit(dataframe, y)

        '''dot_data = tree.export_graphviz(clf, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render(filename="prova", directory='/Users/giuliaciarimboli/Desktop/laurea magistrale/classification trees/graphs')'''

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
                m.add_var_value('a_hat%d_%d' % (i, feat), 1)
                m.add_var_value(('b_%d' % (i)), sk_b[j])
                count += 1

        for t in self.Tb:  # len(skval)
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

    def fit_with_cart(self, dataframe, y):

        sol = self.find_cart(dataframe, y)

        # sol.export('/Users/giuliaciarimboli/Desktop')
        sol.set_time_limit(120)
        sol.solve(log_output=True)
        # sol.print_solution()

        train_error = 0
        for leaf in self.Tl:
            train_error += sol.solution.get_value('L_' + str(leaf))
        train_error = train_error / self.M
        print('train_error:', train_error)

        # GRAPH
        g = pgv.AGraph(directed=True)  # initialize the graph

        nodes = np.append(self.Tb, self.Tl)
        for n in nodes:  # the graph has a node for eache node of the tree
            g.add_node(n, shape='circle', size=8)

            if n != 0:
                father = ceil(n / 2) - 1
                g.add_edge(father, n)

        for t in self.Tb:
            coeff = []
            feat = []
            # if mdl.solution.get_value('d_' + str(t))==0:
            # g.get_node(t).attr['color']='red'
            for f in range(len(self.features)):
                if sol.solution.get_value('a' + str(t) + '_' + str(f)) != 0:
                    coeff.insert(-1, '%.3f' % (sol.solution.get_value('a' + str(t) + '_' + str(f))))
                    feat.insert(-1, f)
            g.get_node(t).attr['label'] = str(coeff) + '*X' + str(feat) + str('<=') + str(
                '%.3f' % (sol.solution.get_value('b_' + str(t))))
        for leaf in self.Tl:
            if sol.solution.get_value('l_' + str(leaf)) == 0:  # these leaves haven't got points
                g.get_node(leaf).attr['color'] = 'red'
        for leaf in self.Tl:
            s = []
            for k in range(len(self.classes)):
                s.append(round(sol.solution.get_value('Nkt_' + str(k) + '_' + str(leaf))))
            for k in range(len(self.classes)):
                if sol.solution.get_value('c_' + str(k) + '_' + str(leaf)) == 1:
                    g.get_node(leaf).attr['label'] = str(s) + '\\n' + 'class %d' % (self.classes[k])
        g.layout(prog='dot')
        g.draw('/Users/giuliaciarimboli/Desktop/w.pdf')

        return sol

    def warm_start(self, dataframe, y, d, modello):

        ordine_l = [0, 1, 4, 3, 10, 9, 8, 6, 22, 21, 20, 19, 18, 17, 16, 15]
        ordine_r = [0, 2, 6, 5, 14, 13, 12, 11, 20, 29, 28, 27, 26, 25, 24, 23]

        mm = SolveSolution(modello)

        T = pow(2, (d + 1)) - 1  # nodes number
        floorTb = int(floor(T / 2))  # number of branch nodes
        Tb = np.arange(0, floorTb)  # range branch nodes
        Tl = np.arange(floorTb, T)  # range leaf nodes
        classes = np.unique(y.values)  # possible labels of classification
        lista_leaf = []
        lista_df = []
        lista_y = []
        y = pd.DataFrame(y)
        lista_df.insert(0, dataframe)
        lista_y.insert(0, y)
        for t in range(int((len(Tb) - 1) / 2) + 1):
            yy = lista_y[t]
            df_split1 = []
            df_split2 = []
            y_1 = []
            y_2 = []
            ind = lista_y[t].index
            ind_df = lista_df[t].index
            if len(lista_y[t]) > self.Nmin:
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
                lista_df.insert(1, df_1)
                lista_df.insert(2, df_2)
                lista_y.insert(1, y_1)
                lista_y.insert(2, y_2)
            else:'''
                mdl = self.fit_with_cart(lista_df[t], lista_y[t])
                cl = yy[9].unique()
                cl.sort()
                for f in self.features:
                    mm.add_var_value('a%d_%d' % (ordine_l[t], f), mdl.solution.get_value('a0_%d' % f))
                    mm.add_var_value('a_hat%d_%d' % (ordine_l[t], f), mdl.solution.get_value('a_hat0_%d' % f))
                mm.add_var_value('b_%d' % (ordine_l[t]), mdl.solution.get_value('b_0'))
                mm.add_var_value('d_%d' % (ordine_l[t]), mdl.solution.get_value('d_0'))
                if 2 * ordine_l[t] + 1 in Tl:
                    leaf = 2 * ordine_l[t] + 1
                    mm.add_var_value('Nt_%d' % leaf, mdl.solution.get_value('Nt_1'))
                    mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_1'))

                    for k in range(len(cl)):
                        mm.add_var_value('c_%d_%d' % (list(classes).index(cl[k]), leaf),
                                         mdl.solution.get_value('c_%d_1' % (k)))
                        mm.add_var_value('Nkt_%d_%d' % (list(classes).index(cl[k]), leaf),
                                         mdl.solution.get_value('Nkt_%d_1' % (k)))
                    kl = list(set(classes) - set(cl))
                    for k2 in range(len(kl)):
                        mm.add_var_value('c_%d_%d' % (list(classes).index(kl[k2]), leaf), 0)
                        mm.add_var_value('Nkt_%d_%d' % (list(classes).index(kl[k2]), leaf), 0)
                    for n in range(len(lista_df[t])):
                        mm.add_var_value('z_%d_%d' % (ind_df[n], leaf), mdl.solution.get_value('z_%d_1' % n))
                    ind_miss = list(set(ind_df) - set(list(np.array(np.arange(0, len(dataframe))))))
                    for n in ind_miss:
                        mm.add_var_value('z_%d_%d' % (n, leaf), 0)

                if 2 * ordine_l[t] + 2 in Tl:
                    leaf = 2 * ordine_l[t] + 2
                    mm.add_var_value('Nt_%d' % leaf, mdl.solution.get_value('Nt_2'))
                    mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_2'))
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
                    if mdl.solution.get_value('z_%d_1' % (i)) == 1:
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
            if len(lista_y_r[t]) > self.Nmin:
                '''for f in self.features:
                    mm.add_var_value('a%d_%d' % (ordine_l[t], f), 0)
                    mm.add_var_value('a_hat%d_%d' % (ordine_l[t], f), 0)
                mm.add_var_value('b_%d' % (ordine_l[t]), 0)
                mm.add_var_value('d_%d' % (ordine_l[t]), 0)
                if 2 * ordine_l[t] + 1 in Tl:
                    leaf = 2 * ordine_l[t] + 1
                    for l in range(leaf, leaf + 2):
                        print(l)
                        mm.add_var_value('Nt_%d' % l, 0)
                        mm.add_var_value('l_%d' % l, 0)
                        for k in range(len(classes)):
                            mm.add_var_value('c_%d_%d' % (k, l),
                                             0)
                            mm.add_var_value('Nkt_%d_%d' % (k, l),
                                             0)
                        for n in range(0, len(dataframe)):
                            mm.add_var_value('z_%d_%d' % (n, l), 0)
                lista_df_r.insert(1, df_1)
                lista_df_r.insert(2,df_2)
                lista_y_r.insert(1, y_1)
                lista_y_r.insert(2, y_2)
            else:'''
                mdl = self.fit_with_cart(lista_df_r[t], lista_y_r[t])
                cl = yy[9].unique()
                cl.sort()

                for f in self.features:
                    mm.add_var_value('a%d_%d' % (ordine_r[t], f), mdl.solution.get_value('a0_%d' % (f)))
                    mm.add_var_value('a_hat%d_%d' % (ordine_r[t], f), mdl.solution.get_value('a_hat0_%d' % (f)))
                mm.add_var_value('b_%d' % (ordine_r[t]), mdl.solution.get_value('b_0'))
                mm.add_var_value('d_%d' % (ordine_r[t]), mdl.solution.get_value('d_0'))
                if 2 * ordine_r[t] + 1 in Tl:
                    leaf = 2 * ordine_r[t] + 1
                    mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_1'))
                    mm.add_var_value('Nt_%d' % (leaf), mdl.solution.get_value('Nt_1'))

                    for k in range(len(cl)):
                        mm.add_var_value('c_%d_%d' % (list(classes).index(cl[k]), leaf),
                                         mdl.solution.get_value('c_%d_1' % (k)))
                        mm.add_var_value('Nkt_%d_%d' % (list(classes).index(cl[k]), leaf),
                                         mdl.solution.get_value('Nkt_%d_1' % (k)))
                    kl = list(set(classes) - set(cl))
                    for k2 in range(len(kl)):
                        mm.add_var_value('c_%d_%d' % (list(classes).index(kl[k2]), leaf), 0)
                        mm.add_var_value('Nkt_%d_%d' % (list(classes).index(kl[k2]), leaf), 0)
                    mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_1'))
                    for n in range(len(lista_df_r[t])):
                        mm.add_var_value('z_%d_%d' % (ind_df[n], leaf), mdl.solution.get_value('z_%d_1' % n))
                    ind_miss = list(set(ind_df) - set(list(np.array(np.arange(0, len(dataframe))))))
                    for n in ind_miss:
                        mm.add_var_value('z_%d_%d' % (n, leaf), 0)
                if 2 * ordine_r[t] + 2 in Tl:
                    leaf = 2 * ordine_r[t] + 2
                    mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_2'))
                    mm.add_var_value('Nt_%d' % (leaf), mdl.solution.get_value('Nt_2'))
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
                        mm.add_var_value('z_%d_%d' % (ind_df[n], leaf), mdl.solution.get_value('z_%d_2' % (n)))
                    ind_miss = list(set(ind_df) - set(list(np.array(np.arange(0, len(dataframe))))))
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

            # GRAPH WARM START
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
            g.draw('/Users/giuliaciarimboli/Desktop/warm_start_LDA.pdf' )

        print('la soluzione warm start:', mm)

        print(mm.check_as_mip_start())
        modello.add_mip_start(mm)

        modello.set_time_limit(900)
        modello.parameters.emphasis.mip = 4

        s = modello.solve(log_output=True)
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
        g.draw('/Users/giuliaciarimboli/Desktop/solfinale_LDA.pdf')

        return a_test, b_test, c_test, train_error

    def test(self, dataframe, y, d, modello, dataframe_test, y_test):

        T = pow(2, (d + 1)) - 1  # nodes number
        floorTb = int(floor(T / 2))  # number of branch nodes
        Tb = np.arange(0, floorTb)  # range branch nodes
        Tl = np.arange(floorTb, T)  # range leaf nodes
        classes = np.unique(y.values)  # possible labels of classification

        a_test, b_test, c_test, train_error = self.warm_start(dataframe, y, d, modello)
        leaves=[]
        prediction = []
        apply= np.zeros( (len(dataframe_test), d+1), dtype=np.int8)
        for p in range(len(dataframe_test)):
            for i in range(d):
                j = int(apply[p][i])
                if np.dot(a_test[j], dataframe_test.loc[p]) < b_test[j]:
                    apply[p][i+1] = 2 * j + 1
                else:
                    apply[p][i+1] = 2 * j + 2
            leaves.insert(p, apply[p][d])

        print(apply)
        count = 0
        for p in leaves:
            leaf = list(Tl).index(p)
            for k in range(len(classes)):
                if c_test[leaf][k] == 1:
                    prediction.insert(count, classes[k])
                    count+=1

        print(leaves)
        print(prediction)
        print(list(y_test))
        errors = 0
        for p in range(len(dataframe_test)):
            if prediction[p] != list(y_test)[p]:
                errors+=1
        test_error = errors/len(y_test)
        print('test error:', test_error)
        print('train error:', train_error)
        return


    def warm_start_univariate(self, dataframe, y, univariate):
        print('risolvo il modello OCT')
        univariate.solve()
        mdl = self.model(dataframe, dataframe, y)
        m = SolveSolution(mdl)
        points = np.arange(0, len(dataframe))
        for t in self.Tb:
            for f in self.features:
                m.add_var_value('a' + str(t) + '_' + str(f), univariate.solution.get_value('a' + str(t) + '_' + str(f)))
                m.add_var_value('a_hat' + str(t) + '_' + str(f),
                                univariate.solution.get_value('a' + str(t) + '_' + str(f)))
                m.add_var_value('s' + str(t) + '_' + str(f), univariate.solution.get_value('a' + str(t) + '_' + str(f)))
        for t in self.Tb:
            m.add_var_value('b_' + str(t), univariate.solution.get_value('b_' + str(t)))
            m.add_var_value('d_' + str(t), univariate.solution.get_value('d_' + str(t)))
        for leaf in self.Tl:
            m.add_var_value('l_' + str(leaf), univariate.solution.get_value('l_' + str(leaf)))
            m.add_var_value('Nt_' + str(leaf), univariate.solution.get_value('Nt_' + str(leaf)))
            m.add_var_value('L_' + str(leaf), univariate.solution.get_value('L_' + str(leaf)))

            for k in range(len(self.classes)):
                m.add_var_value('c_' + str(k) + '_' + str(leaf),
                                univariate.solution.get_value('c_' + str(k) + '_' + str(leaf)))
                m.add_var_value('Nkt_' + str(k) + '_' + str(leaf),
                                univariate.solution.get_value('Nkt_' + str(k) + '_' + str(leaf)))
        for n in points:
            for leaf in self.Tl:
                m.add_var_value('z_' + str(n) + '_' + str(leaf),
                                univariate.solution.get_value('z_' + str(n) + '_' + str(leaf)))
        mdl.add_mip_start(m)
        mdl.solve(log_output=True)

        return mdl


# to use wine dataset

df = pd.read_csv('wine.data.csv', header=None)
y = df[0]
df = df[df.columns[1:]]
#data = df[df.columns[1:]].values
df2 = df
# to use fertility dataset'''

'''df = pd.read_csv('primary-tumor.csv', header=None)
y = df[0]
df = df[df.columns[1:]]
data = df[df.columns[1:]].values
df2 = df'''

# to use thoracic
'''df = pd.read_csv('ThoraricSurgery.csv', header=None)
y = df[16]
df = df[df.columns[0:16]]
data = df[df.columns[0:16]].values
df2 = df'''

'''df = pd.read_csv('car.csv', header=None)
y=df[6]
df=df[df.columns[0:6]]
data= df[df.columns[0:6]].values
df2=df'''

'''df = pd.read_csv('tae.csv', header=None)
y=df[5]
df=df[df.columns[1:5]]
data= df[df.columns[1:5]].values
df2=df'''

# to use prova dataset

'''df = pd.read_csv('ionosphere.csv', header=None)
y = df[34]
df = df[df.columns[0:34]]
data = df[df.columns[0:34]].values
df2 = df'''

df = pd.read_csv('car.csv', header=None)
y = df[6]
df = df[df.columns[0:6]]
data = df[df.columns[0:6]].values
df2 = df

df = pd.read_csv('cmc.csv', header=None)
y = df[9]
df = df[df.columns[0:9]]
data = df[df.columns[0:9]].values
df2 = df

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=2)

# DATA BETWEEN 0-1
scaler = MinMaxScaler()
df_scaled = scaler.fit(X_train)  # save object fitted only with train data
df = scaler.transform(X_train)
df = pd.DataFrame(df)  # scaled dataframe
# y_train = pd.DataFrame(y_train)
# print(y_train)
# print(df)
df_test = scaler.transform(X_test)  # apply same transformation to test set
for i in range(len(df_test)):
    for j in range(len(df_test[0])):
        if df_test[i][j] > 1:
            df_test[i][j] = 1
        elif df_test[i][j] < 0:
            df_test[i][j] = 0
df_test = pd.DataFrame(df_test)

d = 2
a = 0.5
N = int(3/100*(len(df)+len(df_test)))

t = OptimalTree(depth=d, alpha=a, Nmin=N)
#f2 = t.fit_with_cart(df,  y_train)
modello = t.model(df, y_train)

warm = OptimalTree(depth=1, alpha=a, Nmin=N)
warm.test(df, y_train, d, modello, df_test, y_test)
# to fit with cart as warm start
# f2 = t.fit_with_cart(df,  y_train)'''

'''from dec_tree import OptimalTree
ot = OptimalTree(depth=d, alpha=a, Nmin=N)
uni = ot.find_cart(df,  y_train)
f = t.warm_start_univariate(df, y_train, uni)'''


