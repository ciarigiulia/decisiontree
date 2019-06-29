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
import graphviz
from sklearn import tree


class optimaltree(BaseEstimator):

    def __init__(self, depth=2, alpha=0.0, Nmin=0, mipstart = 'CART', dataset='wine.data.csv'):

        self.depth = depth
        self.alpha = alpha
        self.Nmin = Nmin
        self.mipstart = mipstart
        self.dataset = dataset
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
        Tb = np.arange(0, self.floorTb)  # range branch nodes
        Tl = np.arange(self.floorTb, T)  # range leaf nodes

        classes = np.unique(y.values)  # possible labels of classification
        features = dataframe.columns.values  # array of features
        return Tb, Tl, classes, features

    def find_eps(self, dataframe2):
        eps = np.zeros(len(dataframe2.columns))
        for i in range(0, len(dataframe2.columns)):

            if len(dataframe2) >= 2:
                vect = dataframe2[dataframe2.columns[i]].values
                newvect = dataframe2[dataframe2.columns[i]].values

                vect.sort()
                newvect.sort()
                vect = np.delete(vect, 0)
                print(i, vect)

                newvect = np.delete(newvect, len(newvect) - 1)
                print(i, newvect)

                diff = vect - newvect
                diff2 = diff
                count = 0
                for j in range(len(diff)):
                    if diff[j] == 0:
                        diff2 = np.delete(diff2, j - count)
                        count += 1
                if len(diff2)>0:
                    eps[i] = min(diff2)
                else:
                    eps[i] = 1e-5
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
        np.random.seed(1)
        # initialize the model

        mdl = Model('OCT')
        mdl.clear()
        # points = np.array(np.arange(0, len(dataframe)))  # array of point's indexes
        points = dataframe.index

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
            a.append(mdl.binary_var_list(len(self.features), name='a%d' % (t)))  # 'feature_in_node%d'%(t)

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
                        np.dot(dataframe.loc[p1] + self.eps - err * np.ones(len(self.features)), a[m]) + err <= b[
                            m] + self.M1 * (1 - z[p, le + self.floorTb]))

        for p in range(len(points)):
            mdl.add_constraint(mdl.sum(z[p, le + self.floorTb] for le in range(len(self.Tl))) == 1)  #

        for le in range(len(self.Tl)):
            for p in range(len(points)):
                mdl.add_constraint(z[p, le + self.floorTb] <= l[le])

        for le in range(len(self.Tl)):
            mdl.add_constraint(l[le] * self.Nmin <= mdl.sum(z[p, le + self.floorTb] for p in range(len(points))))

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

        # vincolo giorgio
        print('k',self.classes)
        if len(self.classes) <= len(self.Tl):
            for k in range(len(self.classes)):
                mdl.add_constraint(1 <= mdl.sum(c[k, le + self.floorTb] for le in range(len(self.Tl))))

        # vincolo sul massimo numero di foglie associate alle classi
        # mdl.add_constraint(mdl.sum(l[leaf] for leaf in range(len(self.Tl))) == len(self.classes)) #vincolo prof
        if len(self.classes) <= len(self.Tl):
            mdl.add_constraint(mdl.sum(l[leaf] for leaf in range(len(self.Tl))) == len(
                self.Tl))  # questo è utile se il numero di classi supera il numero di foglie

        # OBJECTIVE FUNCTION
        mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * mdl.sum(d[t] for t in self.Tb))

        # mdl.minimize(mdl.sum(L[le]*randL[le] for le in range(len(self.Tl))) + self.alpha * mdl.sum(d[t]*randa[t] for t in self.Tb))
        mdl.print_information()

        return mdl

    def find_cart_warmstart(self, dataframe, dataframe2, y):

        mdl = self.model(dataframe, dataframe2, y)
        # MIP START

        clf = DecisionTreeClassifier(max_depth=self.depth, min_samples_leaf=self.Nmin, random_state=1)
        clf.fit(dataframe, y)

        dot_data = tree.export_graphviz(clf, out_file=None)
        graph = graphviz.Source(dot_data)
        # graph.render(filename="prova",directory='/Users/giuliaciarimboli/Desktop/laurea magistrale/classification trees/graphs',view=True)

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
                m.add_var_value(('b_%d' % (i)), sk_b[j])
                count += 1

        for t in self.Tb:  # len(skval)
            if sk_features[t] >= 0:
                i = list(idx).index(t)
                m.add_var_value(('d_%d' % (i)), 1)
        for leaf in self.Tl:
            m.add_var_value(('l_%d' % (leaf)), 1)

        jj = -1
        for node in idx:
            jj += 1
            k = np.argmax(sk_val[jj][0])
            num = np.sum(sk_val[jj][0])
            ii = list(idx).index(jj)
            if ii in self.Tl:
                m.add_var_value('c_%d_%d' % (k, ii), 1)
                m.add_var_value('Nt_%d' % (ii), num)
                for kl in range(len(self.classes)):
                    m.add_var_value('Nkt_%d_%d' % (kl, ii), sk_val[jj][0][kl])
        missing = len(np.append(self.Tb, self.Tl)) - len(idx)
        for data in range(len(dataframe)):
            foglia = list(idx).index(sk_z[data]) + missing
            m.add_var_value('z_%d_%d' % (data, foglia), 1)
        print(m.check_as_mip_start())
        print(m)
        mdl.add_mip_start(m)

        return mdl

    def find_oct_warmstart(self, dataframe, dataframe2, y):
        mod = self.find_cart_warmstart(dataframe, dataframe2, y)
        mod.set_time_limit(120)
        s = mod.solve(log_output=True)
        mod.print_solution()
        self.draw_graph(s)
        print('find cart solution:')

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
            print(node, 'node')
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
        print('a',a_oct,'b', b_oct,'d', d_oct,'l', l_oct, 'c', c_oct,'z', z_oct,'Nt', Nt_oct, 'Nkt', Nkt_oct)
        return a_oct, b_oct, d_oct, l_oct, c_oct, z_oct, Nt_oct, Nkt_oct

    def fit_with_cart(self, dataframe, dataframe2, y):

        sol = self.find_cart_warmstart(dataframe, dataframe2, y)
        sol.set_time_limit(30)
        sol.export('/Users/giuliaciarimboli/Desktop')
        s = sol.solve(log_output=True)
        print('cart solution')
        sol.print_solution()

        self.draw_graph(s)

        train_error = 0
        for leaf in self.Tl:
            train_error += sol.solution.get_value('L_' + str(leaf))
        train_error = train_error / self.M
        print('train_error:', train_error)

        return sol

    def draw_graph(self, sol):
        g = pgv.AGraph(directed=True)  # initialize the graph

        nodes = np.append(self.Tb, self.Tl)
        for n in nodes:  # the graph has a node for eache node of the tree
            g.add_node(n)
            if n != 0:
                father = ceil(n / 2) - 1
                g.add_edge(father, n)
        for t in self.Tb:
            # if mdl.solution.get_value('d_' + str(t))==0:
            # g.get_node(t).attr['color']='red'
            for f in range(len(self.features)):
                if sol.get_value('a' + str(t) + '_' + str(f)) == 1:
                    g.get_node(t).attr['label'] = str('X[%d]' % (f)) + str('<=') + str(
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
        g.draw('/Users/giuliaciarimboli/Desktop/%s_%s.pdf'%(self.dataset, self.mipstart))

        return g

    def fit_with_oct_mip_start(self, dataframe, dataframe2, y, warm_start):

        sol = self.model(dataframe, dataframe2, y)
        s = SolveSolution(sol)

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
                print(j, k)
                s.add_var_value('Nkt_%d_%d' % (k, leaf), warm_start[7][j][k])

            j += 1
        print(s)
        print(s.check_as_mip_start())
        sol.add_mip_start(s)
        sol.set_time_limit(30)
        # mdl.parameters.mip.tolerances.mipgap(0.1)
        #sol.parameters.emphasis.mip = 4
        print('finding solution with OCT as MIP START:')
        s = sol.solve(log_output=True)

        # sol.print_solution()
        train_error = 0
        for leaf in self.Tl:
            train_error += s.get_value('L_' + str(leaf))
        train_error = train_error / self.M
        print('train_error:', train_error)
        # GRAPH
        self.draw_graph(s)

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


    def warm_start_multi(self, dataframe, dataframe2, y, d, modello):

        ordine_l = [0, 1, 4, 3, 10, 9, 8, 7]
        ordine_r = [0, 2, 6, 5, 14, 13, 12, 11]

        mm = SolveSolution(modello)

        T = pow(2, (d + 1)) - 1  # nodes number
        floorTb = int(floor(T / 2))  # number of branch nodes
        Tb = np.arange(0, floorTb)  # range branch nodes
        Tl = np.arange(floorTb, T)  # range leaf nodes
        classes = np.unique(y.values)  # possible labels of classification

        lista_df = []
        lista_y = []
        y = pd.DataFrame(y)
        lista_df.insert(0, dataframe)
        lista_y.insert(0, y)
        for t in range(int((len(Tb) - 1) / 2) + 1):
            yy = lista_y[t]
            print(yy)
            df_split1 = []
            df_split2 = []
            y_1 = []
            y_2 = []
            ind = lista_y[t].index
            ind_df = lista_df[t].index
            mdl = self.fit_with_cart(lista_df[t], lista_df[t], lista_y[t])
            cl = yy[0].unique()

            cl.sort()
            print(cl)
            print(classes)
            for f in self.features:
                mm.add_var_value('a%d_%d' % (ordine_l[t], f), mdl.solution.get_value('a0_%d' % (f)))
            mm.add_var_value('b_%d' % (ordine_l[t]), mdl.solution.get_value('b_0'))
            mm.add_var_value('d_%d' % (ordine_l[t]), mdl.solution.get_value('d_0'))
            if 2 * ordine_l[t] + 1 in Tl:
                kl = classes
                leaf = 2 * ordine_l[t] + 1
                mm.add_var_value('Nt_%d' % leaf, mdl.solution.get_value('Nt_1'))
                mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_1'))

                for k in range(len(cl)):
                    print(k, cl[k], list(classes).index(cl[k]))
                    mm.add_var_value('c_%d_%d' % (list(classes).index(cl[k]), leaf),
                                     mdl.solution.get_value('c_%d_1' % (k)))
                    mm.add_var_value('Nkt_%d_%d' % (list(classes).index(cl[k]), leaf),
                                     mdl.solution.get_value('Nkt_%d_1' % (k)))
                for k1 in range(len(cl)):
                    list(kl).remove(cl[k1])
                # for k2 in range(len(kl)):
                #    mm.add_var_value('c_%d_%d'%(list(classes).index(kl[k2]), leaf), 0)
                for n in range(len(lista_df[t])):
                    mm.add_var_value('z_%d_%d' % (n, leaf), mdl.solution.get_value('z_%d_1' % (n)))

            if 2 * ordine_l[t] + 2 in Tl:
                kl = classes
                leaf = 2 * ordine_l[t] + 2
                mm.add_var_value('Nt_%d' % leaf, mdl.solution.get_value('Nt_2'))
                mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_2'))
                for k in range(len(cl)):
                    mm.add_var_value('c_%d_%d' % (list(classes).index(cl[k]), leaf),
                                     mdl.solution.get_value('c_%d_2' % (k)))
                    mm.add_var_value('Nkt_%d_%d' % (list(classes).index(cl[k]), leaf),
                                     mdl.solution.get_value('Nkt_%d_2' % (k)))
                for k1 in range(len(cl)):
                    list(kl).remove(cl[k1])
                # for k2 in range(len(kl)):
                #    mm.add_var_value('c_%d_%d'%(list(classes).index(kl[k2]), leaf), 0)
                for n in range(len(lista_df[t])):
                    mm.add_var_value('z_%d_%d' % (n, leaf), mdl.solution.get_value('z_%d_2' % (n)))

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
            print(yy)
            df_split1 = []
            df_split2 = []
            y_1 = []
            y_2 = []
            ind = lista_y_r[t].index
            ind_df = lista_df_r[t].index
            mdl = self.fit_with_cart(lista_df_r[t], lista_df_r[t], lista_y_r[t])
            cl = yy[0].unique()
            cl.sort()

            for f in self.features:
                mm.add_var_value('a%d_%d' % (ordine_r[t], f), mdl.solution.get_value('a0_%d' % (f)))

            mm.add_var_value('b_%d' % (ordine_r[t]), mdl.solution.get_value('b_0'))
            mm.add_var_value('d_%d' % (ordine_r[t]), mdl.solution.get_value('d_0'))
            if 2 * ordine_r[t] + 1 in Tl:
                kl = classes
                leaf = 2 * ordine_r[t] + 1
                mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_1'))
                mm.add_var_value('Nt_%d' % (leaf), mdl.solution.get_value('Nt_1'))

                for k in range(len(cl)):
                    mm.add_var_value('c_%d_%d' % (list(classes).index(cl[k]), leaf),
                                     mdl.solution.get_value('c_%d_1' % (k)))
                    mm.add_var_value('Nkt_%d_%d' % (list(classes).index(cl[k]), leaf),
                                     mdl.solution.get_value('Nkt_%d_1' % (k)))
                for k1 in range(len(cl)):
                    list(kl).remove(cl[k1])
                # for k2 in range(len(kl)):
                #    mm.add_var_value('c_%d_%d'%(list(classes).index(kl[k2], leaf)), 0)
                mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_1'))
                for n in range(len(lista_df_r[t])):
                    mm.add_var_value('z_%d_%d' % (n, leaf), mdl.solution.get_value('z_%d_1' % (n)))
            if 2 * ordine_r[t] + 2 in Tl:
                kl = classes
                leaf = 2 * ordine_r[t] + 2
                mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_2'))
                mm.add_var_value('Nt_%d' % (leaf), mdl.solution.get_value('Nt_2'))

                for k in range(len(cl)):
                    mm.add_var_value('c_%d_%d' % (list(classes).index(cl[k]), leaf),
                                     mdl.solution.get_value('c_%d_2' % (k)))
                    mm.add_var_value('Nkt_%d_%d' % (list(classes).index(cl[k]), leaf),
                                     mdl.solution.get_value('Nkt_%d_2' % (k)))
                for k1 in range(len(cl)):
                    list(kl).remove(cl[k1])
                # for k2 in range(len(kl)):
                #    mm.add_var_value('c_%d_%d'%(list(classes).index(kl[k2]), leaf), 0)
                for n in range(len(lista_df_r[t])):
                    mm.add_var_value('z_%d_%d' % (n, leaf), mdl.solution.get_value('z_%d_2' % (n)))

            for i in range(len(lista_df_r[t])):
                j = ind[i]
                m = ind_df[i]
                if mdl.solution.get_value('z_%d_1' % (i)) == 1:
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
            g.draw('/Users/giuliaciarimboli/Desktop/warm_start.pdf')

        print('la soluzione warm start:', mm)

        print(mm.check_as_mip_start())
        modello.add_mip_start(mm)
        modello.set_time_limit(3600)

        modello.solve(log_output=True)
        modello.print_solution()

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
        g.draw('/Users/giuliaciarimboli/Desktop/sol finale.pdf')

        return modello

    def test_model(self, dataframe, dataframe2, y):
        err = min(self.eps)

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
                    mdl.add_constraint(
                        np.dot(dataframe.loc[p] + self.eps - err * np.ones(len(self.features)), self.A[m]) + err <=
                        self.B[m] + self.M1 * (
                                1 - z[p, le + self.floorTb]))

        for p in points:
            mdl.add_constraint(
                mdl.sum(
                    z[p, le + self.floorTb] for le in range(len(self.Tl))) == 1)  # each poin associate to a leaf

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

    def test_model(self, dataframe, dataframe2, y):
        # TODO fare test solo con a e b e costruire funzione apply
        err = min(self.eps)

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
                    mdl.add_constraint(
                        np.dot(dataframe.loc[p] + self.eps - err * np.ones(len(self.features)), self.A[m]) + err <=
                        self.B[m] + self.M1 * (
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


# to use wine dataset
'''df = pd.read_csv('ionosphere.csv', header=None)
y = df[0] - 1
df = df[df.columns[1:]]
data = df[df.columns[1:]].values
df2 = df'''
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

'''df = pd.read_csv('prova.csv', header=None)
y = df[2]
df = df[df.columns[0:2]]
data = df[df.columns[0:2]].values
df2 = df'''

'''X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0, random_state=1)

# DATA BETWEEN 0-1
scaler = MinMaxScaler()
df_scaled = scaler.fit(X_train)  # save object fitted only with train data
df_scaled = scaler.transform(X_train)
df = pd.DataFrame(df_scaled)  # scaled dataframe
df2 = scaler.transform(X_train)
df2 = pd.DataFrame(df2)'''

'''df_test = scaler.transform(X_test)  # apply same transformation to test set
for i in range(len(df_test)):
    for j in range(len(df_test[0])):
        if df_test[i][j] > 1:
            df_test[i][j] = 1
        elif df_test[i][j] < 0:
            df_test[i][j] = 0
df_test = pd.DataFrame(df_test)
df_test2 = scaler.transform(X_test)
df_test2 = pd.DataFrame(df_test2)'''

d = 2
a = 0.5
N = 1
datasets = ['wine.data.csv']
data = datasets[0]
mipstarts = ['OCT']
for dataset in datasets:
    df = pd.read_csv(dataset, header=None)
    y = df[0]
    df = df[df.columns[1:]]
    data = df[df.columns[1:]].values
    df2 = df
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=1)

    # DATA BETWEEN 0-1
    scaler = MinMaxScaler()
    df_scaled = scaler.fit(X_train)  # save object fitted only with train data
    df_scaled = scaler.transform(X_train)
    df = pd.DataFrame(df_scaled)  # scaled dataframe
    df2 = scaler.transform(X_train)
    df2 = pd.DataFrame(df2)
    print(df)

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

    for m in mipstarts:
        t = optimaltree(depth= d, alpha=a, Nmin=N, mipstart=m, dataset=dataset)
        if m == 'CART':
            print('CART')
            f2 = t.fit_with_cart(df, df2, y_train)
        elif m == 'OCT':
            warm = optimaltree(depth=d-1,alpha=a, Nmin=N, mipstart=m, dataset=dataset)
            ws_ = warm.find_oct_warmstart(df, df2, y_train)
            f = t.fit_with_oct_mip_start(df, df2, y_train, ws_)

#
#modello = t.model(df, df2, y_train)

#warm = optimaltree(depth=1, alpha=a, Nmin=N)
#ws = warm.warm_start_multi(df, df2, y_train, d, modello)
# predict = t.test_model(df_test, df_test2, y_test)'''
