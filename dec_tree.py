from tree import tree

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler




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



t = tree(depth=3, alpha=0.5, Nmin=1)
f = t.fit(df, df2, y_train)
predict = t.test_model(df_test, df_test2, y_test)
