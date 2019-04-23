from math import floor
from math import ceil
from time import time

import numpy as np
import pandas as pd
from docplex.mp.model import Model
from docplex.mp.context import Context
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from docplex.mp.solution import SolveSolution
import pygraphviz as pgv
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn import tree

class optimaltree(BaseEstimator):

    def __init__(self, depth=2, alpha=0.0, Nmin=1, max_features=200):

        self.depth = depth
        self.alpha = alpha
        self.Nmin = Nmin
        self.max_features = max_features

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

    def model(self, dataframe, dataframe2, y):
        self.Tb, self.Tl, self.classes, self.features = self.find_T(dataframe, y)
        self.parent = self.find_pt()
        self.Al, self.Ar = self.find_anc()

        np.random.seed(1)
        # initialize the model
        self.M = len(dataframe)
        mdl = Model(name='OCT-H')
        #points = np.array(np.arange(0, len(dataframe)))  # array of point's indexes
        points = dataframe.index
        Num_points = len(dataframe)  # number of points

        # define Y matrix
        Y = np.arange(len(self.classes) * len(points)).reshape(len(points), len(self.classes))
        for i in range(0, len(points)):
            for k in range(0, len(self.classes)):
                k1 = self.classes[k]
                if y.values[i] == k1:
                    Y[i, k] = 1
                else:
                    Y[i, k] = -1

        # VARIABLES

        # for each branch node associate a feature, 1 if in node t I take feature f
        a = []
        for t in self.Tb:
            a.append(mdl.continuous_var_list(len(self.features), ub=1, lb=-1, name='a%d' % (t)))  # 'feature_in_node%d'%(t)
        a_hat =[]
        for t in self.Tb:
            a_hat.append(mdl.continuous_var_list(len(self.features), name='a_hat%d' % (t))) # TODO va messo upper/lower bound qui?
        s = []
        for t in self.Tb:
            s.append(mdl.binary_var_list(len(self.features), name='s%d' % (t)))
        # for each branch node associate a variable
        b = mdl.continuous_var_list(self.Tb, name='b')  # 'hyperplane_coefficient'

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
                    Nkt[k, le + self.floorTb] == 0.5 * mdl.sum((1 + Y[i, k]) * z[i, le + self.floorTb] for i in range(len(points))))

        for le in range(len(self.Tl)):
            mdl.add_constraint(Nt[le] == mdl.sum(z[p, le + self.floorTb] for p in range(len(points))))

        for le in range(len(self.Tl)):
            mdl.add_constraint(l[le] == mdl.sum(c[k, le + self.floorTb] for k in range(len(self.classes))))

        for p in range(len(points)):
            p1=points[p]
            for le in range(len(self.Tl)):
                for n in self.Ar[le + self.floorTb]:

                    mdl.add_constraint(np.dot(dataframe.loc[p1], a[n]) >= b[n] - self.bigM * (1 - z[p, le + self.floorTb]))

        for p in range(len(points)):
            p1=points[p]
            for le in range(len(self.Tl)):
                for m in self.Al[le + self.floorTb]:
                    mdl.add_constraint(
                        np.dot(dataframe.loc[p1], a[m]) + self.mu <= b[m] + (self.bigM + self.mu) * (1 - z[p, le + self.floorTb]))

        for p in range(len(points)):
            mdl.add_constraint(mdl.sum(z[p, le + self.floorTb] for le in range(len(self.Tl))) == 1)  #

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

        for t in np.delete(self.Tb, 0):
            mdl.add_constraint(d[t] <= d[self.parent[t]])

        randa = [np.random.uniform(1 - 1e-1, 1 + 1e-1) for t in self.Tb]
        randL = [np.random.uniform(1 - 1e-1, 1 + 1e-1) for t in self.Tl]

        for i in range(0, len(self.Tl), 2):
            mdl.add_constraint(l[i] <= d[self.find_pt()[i + self.depth]])

        # vincolo giorgio
        if len(self.classes) <= len(self.Tl):
            for k in range(len(self.classes)):
                mdl.add_constraint(1 <= mdl.sum(c[k, le + self.floorTb] for le in range(len(self.Tl))))

        # vincolo sul massimo numero di foglie associate alle classi
        # mdl.add_constraint(mdl.sum(l[leaf] for leaf in range(len(self.Tl))) == len(self.classes)) #vincolo prof

        #vincolo prendere al massimo un tot di features
        #for t in self.Tb:
        #    mdl.add_constraint(mdl.sum(s[t][f] for f in self.features) <= self.max_features)

        # OBJECTIVE FUNCTION
        #mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))))

        mdl.minimize(mdl.sum(L[le] for le in range(len(self.Tl))) + self.alpha * (mdl.sum(s[t][f] for t in self.Tb for f in self.features)))

        # mdl.minimize(mdl.sum(L[le]*randL[le] for le in range(len(self.Tl))) + self.alpha * mdl.sum(d[t]*randa[t] for t in self.Tb))
        mdl.print_information()

        return mdl

    def find_cart(self, dataframe, dataframe2, y):

        mdl = self.model(dataframe, dataframe2, y)
        # MIP START

        clf = DecisionTreeClassifier(max_depth=self.depth, min_samples_leaf=self.Nmin, random_state=1)
        clf.fit(dataframe, y)

        dot_data = tree.export_graphviz(clf, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render(filename="prova", directory='/Users/giuliaciarimboli/Desktop/laurea magistrale/classification trees/graphs', view=True)

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
                m.add_var_value('s%d_%d' % (i, feat), 1)
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
        missing = len(np.append(self.Tb, self.Tl))-len(idx)
        for data in range(len(dataframe)):
            foglia = list(idx).index(sk_z[data]) + missing
            m.add_var_value('z_%d_%d' % (data, foglia), 1)
        mdl.add_mip_start(m)

        return mdl

    def fit_with_cart(self, dataframe, dataframe2, y):

        sol = self.find_cart(dataframe, dataframe2, y)

        #sol.export('/Users/giuliaciarimboli/Desktop')
        sol.set_time_limit(300)
        sol.solve(log_output=True)
        sol.print_solution()

        train_error = 0
        for leaf in self.Tl:
            train_error += sol.solution.get_value('L_' + str(leaf))
        train_error = train_error / self.M
        print('train_error:', train_error)

        # GRAPH
        g  = pgv.AGraph(directed=True) # initialize the graph

        nodes = np.append(self.Tb, self.Tl)
        for n in nodes: #the graph has a node for eache node of the tree
            g.add_node(n, shape='circle', size=8)

            if n != 0:
                father = ceil(n / 2) - 1
                g.add_edge(father, n)

        for t in self.Tb:
            coeff=[]
            feat=[]
            #if mdl.solution.get_value('d_' + str(t))==0:
                #g.get_node(t).attr['color']='red'
            for f in range(len(self.features)):
                if sol.solution.get_value('a'+str(t)+'_'+str(f)) != 0:
                    coeff.insert(-1, '%.3f'%(sol.solution.get_value('a' + str(t) + '_' + str(f))))
                    feat.insert(-1, f)
            g.get_node(t).attr['label'] = str(coeff)+'*X' + str(feat) + str('<=') + str('%.3f'%(sol.solution.get_value('b_'+str(t))))
        for leaf in self.Tl:
            if sol.solution.get_value('l_' + str(leaf))==0: # these leaves haven't got points
                g.get_node(leaf).attr['color']='red'
        for leaf in self.Tl:
            s = []
            for k in range(len(self.classes)):
                s.append(round(sol.solution.get_value('Nkt_' + str(k) + '_' + str(leaf))))
            for k in range(len(self.classes)):
                if sol.solution.get_value('c_' + str(k) + '_' + str(leaf)) == 1:
                    g.get_node(leaf).attr['label']= str(s) + '\\n' + 'class %d'%(self.classes[k])
        g.layout(prog='dot')
        g.draw('/Users/giuliaciarimboli/Desktop/w.pdf')

        return sol

    def warm_start(self, dataframe, dataframe2, y, d, modello):

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
        for t in range(int((len(Tb)-1)/2)+1):
            yy = lista_y[t]

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
                mm.add_var_value('a%d_%d'%(ordine_l[t], f), mdl.solution.get_value('a0_%d'%(f)))
                mm.add_var_value('a_hat%d_%d' % (ordine_l[t], f), mdl.solution.get_value('a_hat0_%d' % (f)))
                mm.add_var_value('s%d_%d' % (ordine_l[t], f), mdl.solution.get_value('s0_%d' % (f)))
            mm.add_var_value('b_%d'%(ordine_l[t]), mdl.solution.get_value('b_0'))
            mm.add_var_value('d_%d'%(ordine_l[t]), mdl.solution.get_value('d_0'))
            if 2*ordine_l[t]+1 in Tl :
                leaf = 2*ordine_l[t]+1
                mm.add_var_value('Nt_%d'%leaf, mdl.solution.get_value('Nt_1'))
                mm.add_var_value('l_%d'%(leaf), mdl.solution.get_value('l_1'))


                for k in range(len(cl)):
                    print(k,cl[k], list(classes).index(cl[k]))
                    mm.add_var_value('c_%d_%d'%(list(classes).index(cl[k]), leaf), mdl.solution.get_value('c_%d_1'%(k)))
                    mm.add_var_value('Nkt_%d_%d'%(list(classes).index(cl[k]), leaf),mdl.solution.get_value('Nkt_%d_1'%(k)))
                for n in range(len(lista_df[t])):
                    mm.add_var_value('z_%d_%d'%(n, leaf), mdl.solution.get_value('z_%d_1'%(n)))


            if 2 * ordine_l[t] + 2 in Tl:
                leaf = 2 * ordine_l[t] + 2
                mm.add_var_value('Nt_%d'%leaf, mdl.solution.get_value('Nt_2'))
                mm.add_var_value('l_%d' %(leaf), mdl.solution.get_value('l_2'  ))
                for k in range(len(cl)):
                    mm.add_var_value('c_%d_%d' % (list(classes).index(cl[k]), leaf), mdl.solution.get_value('c_%d_2' % (k)))
                    mm.add_var_value('Nkt_%d_%d'%(list(classes).index(cl[k]), leaf),mdl.solution.get_value('Nkt_%d_2'%(k)))

                for n in range(len(lista_df[t])):
                    mm.add_var_value('z_%d_%d' % (n, leaf), mdl.solution.get_value('z_%d_2' % (n)))

            for i in range(len(lista_df[t])):
                j = ind[i]
                m = ind_df[i]
                if mdl.solution.get_value('z_%d_1'%(i)) == 1:
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

        for t in range(1, int((len(Tb)-1)/2)+1):
            yy = lista_y_r[t]

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
                mm.add_var_value('a%d_%d'%(ordine_r[t], f), mdl.solution.get_value('a0_%d'%(f)))
                mm.add_var_value('a_hat%d_%d' % (ordine_r[t], f), mdl.solution.get_value('a_hat0_%d' % (f)))
                mm.add_var_value('s%d_%d' % (ordine_r[t], f), mdl.solution.get_value('s0_%d' % (f)))
            mm.add_var_value('b_%d'%(ordine_r[t]), mdl.solution.get_value('b_0'))
            mm.add_var_value('d_%d'%(ordine_r[t]), mdl.solution.get_value('d_0'))
            if 2*ordine_r[t]+1 in Tl :
                leaf = 2*ordine_r[t]+1
                mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_1'))
                mm.add_var_value('Nt_%d'%(leaf), mdl.solution.get_value('Nt_1'))

                for k in range(len(cl)):
                    mm.add_var_value('c_%d_%d'%(list(classes).index(cl[k]), leaf), mdl.solution.get_value('c_%d_1'%(k)))
                    mm.add_var_value('Nkt_%d_%d'%(list(classes).index(cl[k]), leaf),mdl.solution.get_value('Nkt_%d_1'%(k)))

                mm.add_var_value('l_%d'%(leaf), mdl.solution.get_value('l_1'))
                for n in range(len(lista_df_r[t])):
                    mm.add_var_value('z_%d_%d'%(n, leaf), mdl.solution.get_value('z_%d_1'%(n)))
            if 2 * ordine_r[t] + 2 in Tl:
                leaf = 2 * ordine_r[t] + 2
                mm.add_var_value('l_%d' % (leaf), mdl.solution.get_value('l_2'))
                mm.add_var_value('Nt_%d'%(leaf), mdl.solution.get_value('Nt_2'))

                for k in range(len(cl)):
                    mm.add_var_value('c_%d_%d' % (list(classes).index(cl[k]), leaf), mdl.solution.get_value('c_%d_2' % (k)))
                    mm.add_var_value('Nkt_%d_%d'%(list(classes).index(cl[k]), leaf),mdl.solution.get_value('Nkt_%d_2'%(k)))

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
        
        print('la soluzione warm start:',mm)

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


'''df = pd.read_csv('fertility.csv', header=None)
y=df[9]
df=df[df.columns[0:9]]
data= df[df.columns[0:9]].values
df2=df'''

# to use prova dataset

'''df = pd.read_csv('prova.csv', header=None)
y = df[2]
df = df[df.columns[0:2]]
data = df[df.columns[0:2]].values
df2 = df'''

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0, random_state=1)

# DATA BETWEEN 0-1
scaler = MinMaxScaler()
df_scaled = scaler.fit(X_train)  # save object fitted only with train data
df = scaler.transform(X_train)
df = pd.DataFrame(df)  # scaled dataframe
df2 = scaler.transform(X_train)
df2 = pd.DataFrame(df2)
#y_train = pd.DataFrame(y_train)
#print(y_train)
#print(df)
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
a = 0.1
N = 10
#F = 5 # len(df.columns)

t = optimaltree(depth=d, alpha=a, Nmin=N)
#f2 = t.fit_with_cart(df, df2, y_train)
modello = t.model(df, df2, y_train)

warm = optimaltree(depth=1, alpha=a, Nmin=N)
ws = warm.warm_start(df, df2, y_train, d, modello)

# to fit with cart as warm start
#f2 = t.fit_with_cart(df, df2, y_train)
