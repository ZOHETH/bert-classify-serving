import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re


def tag_sum(x, tags):
    res = ''
    for tag in tags:
        res += x[tag] + '.'
    return res[:-1]


def load_x_y():
    df = pd.read_csv('dataset/data.csv')
    df['小类'] = df.apply(lambda x: tag_sum(x, ('大类', '小类')), axis=1)
    # 筛选数量不够的类别
    count_df = (df.groupby('小类').count())
    tags = list(count_df.loc[count_df['问'] > 100].index)
    df = df.loc[df['小类'].isin(tags)]

    df=df.loc[df['问'].str.len()>4]
    print(len(df))
    X = df['问'].values
    y = df['小类'].values
    # 长句分割
    # res_X=[]
    # res_y=[]
    # for i,s in enumerate(X):
    #     s_list=re.split('[。？]',s)
    #     for res_s in s_list:
    #         res_X.append(res_s)
    #         res_y.append(y[i])
    X=[x[:60] for x in X]
    np.save('X.npy', X)
    np.save('y.npy', y)


def load_data():
    X = np.load('X.npy', allow_pickle=True)
    y = np.load('y.npy', allow_pickle=True)
    print(len(set(y)))
    m = 0
    for i in X:
        m = max(m, len(i))
    print(m)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    load_x_y()
