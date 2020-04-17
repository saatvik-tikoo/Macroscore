from collections import defaultdict
import pandas
from sklearn import ensemble
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras
import sklearn.metrics as mx
from matplotlib import pyplot as plt
import copy
import pickle

class Model:
    def __init__(self, label_type, neural_model, fileName):
        self.label_type = label_type
        self.neural_model = neural_model
        self.fileName = fileName
        self.df = None

    def __score_model__(self, X_train, X_test, y_train, y_test, clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return mx.f1_score(y_test, y_pred)

    def __get_baseline__(self):
        total_true = total_false = total_val = 0
        for idx, row in self.df.iterrows():
            total_val += 1
            if row[self.label_type] == 1:
                total_true += 1
            else:
                total_false += 1
        print("Total rows is: ", total_val)
        print("Total papers reproducible: ", total_true)
        print("total_true % is: ", (total_true / total_val) * 100)
        print("total_false % is:", (total_false / total_val) * 100)

    def get_random_kfolds(self, train_set_size=0.9, seed=0, authors=None):
        if authors is None:
            return
        df = self.df.copy()
        train_set, test_set = pandas.DataFrame(columns=list(df.columns)), pandas.DataFrame(columns=list(df.columns))
        while len(train_set) <= self.df.shape[0] * train_set_size and len(authors) > 0 and df.shape[0] > 0:
            np.random.seed(seed)
            author = authors[np.random.randint(0, len(authors))]
            temp_df = df.loc[df['Authors.O'].str.contains(author, case=False)]
            train_set = pandas.concat([train_set, temp_df], ignore_index=True)
            authors.remove(author)

            # remove all the rows from df that have been added to the new set
            df.drop(df[df['Authors.O'].str.contains(author, case=False)].index, inplace=True)

            if test_set.shape[0] <= self.df.shape[0] * (1 - train_set_size) and len(authors) > 0 and df.shape[0] > 0:
                np.random.seed(seed)
                author = authors[np.random.randint(0, len(authors))]
                temp_df = df.loc[df['Authors.O'].str.contains(author, case=False)]
                test_set = pandas.concat([test_set, temp_df], ignore_index=True)
                authors.remove(author)

                # remove all the rows from df that have been added to the new set
                df.drop(df[df['Authors.O'].str.contains(author, case=False)].index, inplace=True)

        train_set = train_set.drop(['Authors.O'], axis=1)
        test_set = test_set.drop(['Authors.O'], axis=1)
        return train_set, test_set

    def modelling_custom_kfolds(self):
        gnb = GaussianNB()
        neigh = KNeighborsClassifier(n_neighbors=10, p=2, weights='uniform')
        forest = ensemble.RandomForestClassifier(max_depth=7, random_state=0, bootstrap=True)

        result = defaultdict(list)
        authors = []
        for au in list(self.df['Authors.O']):
            for j in au.split(','):
                author = j.strip().lower()
                if author not in authors:
                    authors.append(author)
        for idx in range(10):
            print('Getting {} set '.format(idx + 1))
            train_set, test_set = self.get_random_kfolds(train_set_size=0.8, seed=idx, authors=copy.deepcopy(authors))
            X_train = train_set.drop([self.label_type, 'doi'], axis=1)
            y_train = train_set[self.label_type]
            y_train = y_train.astype('int')

            X_test = test_set.drop([self.label_type, 'doi'], axis=1)
            y_test = test_set[self.label_type]
            y_test = y_test.astype('int')

            gnb.fit(X_train, y_train)
            gnb_pred = gnb.predict(X_test)
            result['gnb'].append(mx.f1_score(y_test, gnb_pred))

            neigh.fit(X_train, y_train)
            neigh_pred = neigh.predict(X_test)
            result['neigh'].append(mx.f1_score(y_test, neigh_pred))

            forest.fit(X_train, y_train)
            forest_pred = forest.predict(X_test)
            result['forest'].append(mx.f1_score(y_test, forest_pred))

        print("Accuracy of Naive Bayes is: %0.2f" % np.mean(result['gnb']))
        print("Accuracy of KNN is: %0.2f" % np.mean(result['neigh']))
        print("Accuracy of Random Forest is: %0.2f" % np.mean(result['forest']))


    def get_data(self):
        self.df = pandas.read_excel(self.fileName, encoding='ansi')
        self.__get_baseline__()


if __name__ == '__main__':
    files = ['data/XeuanFeatures.xlsx']
    for i in range(len(files)):
        print('----------------Results for {} file----------------'.format(files[i].split('.')[0]))
        mscore = Model('label', neural_model=False, fileName=files[i])
        mscore.get_data()
        mscore.modelling_custom_kfolds()
