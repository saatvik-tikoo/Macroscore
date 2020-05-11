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
    def __init__(self, label_type, neural_model, fileName, folds=10):
        self.label_type = label_type
        self.neural_model = neural_model
        self.fileName = fileName
        self.df = None
        self.folds = folds

    def __score_model__(self, X_train, X_test, y_train, y_test, clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return mx.f1_score(y_test, y_pred)

    def ablation_test(self):
        self.__remove_unusable_features__()
        X = self.df.drop([self.label_type], axis=1)
        y = self.df[self.label_type]

        gnb = GaussianNB()
        neigh = KNeighborsClassifier(n_neighbors=6, p=2, weights='uniform')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        base_score = dict()
        base_score['gnb'] = self.__score_model__(X_train, X_test, y_train, y_test, gnb)
        base_score['neigh'] = self.__score_model__(X_train, X_test, y_train, y_test, neigh)

        scores = defaultdict(list)
        for idx in range(X_train.shape[1]):
            cols = [ndx != idx for ndx in range(X_train.shape[1])]
            scores['gnb'].append(self.__score_model__(X_train.iloc[:, cols], X_test.iloc[:, cols], y_train, y_test, gnb))
            scores['neigh'].append(self.__score_model__(X_train.iloc[:, cols], X_test.iloc[:, cols], y_train, y_test, neigh))

        final_scores_gnb = dict()
        final_scores_neigh = dict()
        for k, v in scores.items():
            if k == 'gnb':
                for idx in range(len(v)):
                    final_scores_gnb[X.columns[idx]] = (v[idx] - base_score[k])
                final_scores_gnb = sorted(final_scores_gnb.items(), key=lambda kv: kv[1], reverse=True)
            elif k == 'neigh':
                for idx in range(len(v)):
                    final_scores_neigh[X.columns[idx]] = (v[idx] - base_score[k])
                final_scores_neigh = sorted(final_scores_neigh.items(), key=lambda kv: kv[1], reverse=True)

        print('Based on Naive Bayes: ', final_scores_gnb)
        print('Based on KNN: ', final_scores_neigh)

    def __remove_unusable_features__(self):
        print('Initial Shape is: ', self.df.shape)
        if self.label_type == 'pvalue.label':
            self.df = self.df.drop(['P.value.R', 'Direction.R', 'O.within.CI.R', 'Meta.analysis.significant'], axis=1)
        elif self.label_type == 'O.within.CI.R':
            self.df = self.df.drop(['P.value.R', 'Direction.R', 'Meta.analysis.significant'], axis=1)
        elif self.label_type == 'Meta.analysis.significant':
            self.df = self.df.drop(['P.value.R', 'Direction.R', 'O.within.CI.R'], axis=1)

        cols_drop = {'DOI', '1st.author.O', 'Senior.author.O', 'Study.Title.O', 'Unnamed: 0', 'Unnamed: 0.1', 'Volume.O'}
        cols_total = set(self.df.columns)
        self.df = self.df.drop(cols_drop.intersection(cols_total), axis=1)
        self.df = self.df.dropna()
        print('Shape after data cleaning is: ', self.df.shape)

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

    def select_best_features_chi2(self):
        X = self.df.drop([self.label_type, 'Authors.O'], axis=1)
        cols = X.columns
        X = MinMaxScaler().fit_transform(X)
        y = self.df[self.label_type]
        bestfeatures = SelectKBest(score_func=chi2, k='all')
        fit = bestfeatures.fit(X, y)
        dfscores = pandas.DataFrame(fit.scores_)
        dfcolumns = pandas.DataFrame(cols)
        featureScores = pandas.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']
        final_features = featureScores.nlargest(featureScores.shape[0] - 2, 'Score')
        pandas.set_option('display.max_rows', final_features.shape[0] + 1)
        return final_features

    def train_test_cv_split(self, index):
        test_set = self.df.loc[self.df['Fold_Id'] == index]
        train_set = self.df.loc[self.df.index.difference(test_set.index)]
        return train_set, test_set

    def modelling(self, best_features):
        self.df = self.df.drop(['Authors.O'], axis=1)
        gnb = GaussianNB()
        neigh = KNeighborsClassifier(n_neighbors=6, p=2, weights='uniform')
        forest = ensemble.RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0, bootstrap=True)
        model = None
        if self.neural_model:
            model = keras.Sequential([
                keras.layers.Dense(32, input_dim=len(best_features), activation='sigmoid'),
                keras.layers.Dense(16, activation='sigmoid'),
                keras.layers.Dense(32, activation='sigmoid'),
                keras.layers.Dense(8, activation='sigmoid'),
                keras.layers.Dense(1, activation='sigmoid')
            ])
        result = defaultdict(list)
        for idx in range(self.folds):
            train_set, test_set = self.train_test_cv_split(i + 1)

            X_train = train_set[best_features]
            y_train = train_set[self.label_type]
            y_train = y_train.astype('int')

            X_test = test_set[best_features]
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

            if self.neural_model:
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
                _, accuracy = model.evaluate(X_test, y_test)
                result['neuralNetwork'].append(accuracy)

        print("Accuracy of Naive Bayes is: %0.2f" % np.mean(result['gnb']))
        print("Accuracy of KNN is: %0.2f" % np.mean(result['neigh']))
        print("Accuracy of Random Forest is: %0.2f" % np.mean(result['forest']))
        if self.neural_model:
            print("Accuracy of Neural Network is: %0.2f" % np.mean(result['neuralNetwork']))

    def tuning_hyperparameters(self, best_features):
        res_best = defaultdict(list)

        for idx in range(self.folds):
            print('\n----------Round {}----------'.format(idx + 1))
            train_set, test_set = self.train_test_cv_split(i + 1)

            X_train = train_set[best_features]
            y_train = train_set[self.label_type]
            y_train = y_train.astype('int')

            X_test = test_set[best_features]
            y_test = test_set[self.label_type]
            y_test = y_test.astype('int')

            result = defaultdict(list)
            for n_neighbors_val in range(3, 15):
                for p_val in range(1, 6):
                    neigh = KNeighborsClassifier(n_neighbors=n_neighbors_val, p=p_val, weights='uniform')
                    neigh.fit(X_train, y_train)
                    neigh_pred = neigh.predict(X_test)
                    result[(n_neighbors_val, p_val)].append(np.round(mx.f1_score(y_test, neigh_pred), 2))
            final_val = max(result.items(), key=lambda x: x[1])
            print("Accuracy of KNN is: {}".format(final_val))
            res_best['knn'].append(final_val)

            result = defaultdict(list)
            for max_depth in range(2, 11):
                forest = ensemble.RandomForestClassifier(random_state=0, n_estimators=100, max_depth=max_depth,
                                                         bootstrap=True)
                forest.fit(X_train, y_train)
                forest_pred = forest.predict(X_test)
                result[(100, max_depth)].append(np.round(mx.f1_score(y_test, forest_pred), 2))
            final_val = max(result.items(), key=lambda x: x[1])
            print("Accuracy of Random forest is: {}".format(final_val))
            res_best['forest'].append(final_val)

        print('----------Final Values----------')
        aggregate_val = defaultdict(list)
        for k, v in res_best.items():
            aggregate_val[k].extend([0, 0])
            total_val = 0
            for val in v:
                aggregate_val[k][0] += val[0][0] * val[1][0]
                aggregate_val[k][1] += val[0][1] * val[1][0]
                total_val += val[1][0]
            aggregate_val[k][0] /= total_val
            aggregate_val[k][1] /= total_val
            print('Best Possible Values for {} are: {}'.format(k, aggregate_val[k]))

    def get_data(self):
        self.df = pandas.read_excel(self.fileName, encoding='ansi')
        self.__remove_unusable_features__()
        self.__get_baseline__()

def mergeAllFeatures(f_read, f_write):

    mscore_t = Model('Meta.analysis.significant', neural_model=False, fileName=f_read[0])
    mscore_t.get_data()
    features_t = mscore_t.select_best_features_chi2()
    b_features_t = list(features_t['Specs'])[: 10]
    res = mscore_t.df[b_features_t]
    res = pandas.concat([res, mscore_t.df[['DOI', 'P.value.R', 'Direction.R', 'O.within.CI.R', 'Meta.analysis.significant', 'Authors.O']]], axis=1)

    for f_idx in range(1, len(f_read)):
        mscore_t = Model('Meta.analysis.significant', neural_model=False, fileName=f_read[f_idx])
        mscore_t.get_data()
        features_t = mscore_t.select_best_features_chi2()
        b_features_t = list(features_t['Specs'])[: 10]
        res = pandas.concat([res, mscore_t.df[b_features_t]], axis=1)
    res.to_excel(f_write)
    print(res.columns)


if __name__ == '__main__':
    all_labels = ['pvalue.label', 'O.within.CI.R', 'Meta.analysis.significant']
    files = ['data/final_references_network_2hops_wos_synthetic_1.xlsx',
             'data/final_references_network_2hops_mag_synthetic_1.xlsx',
             'data/final_citations_network_2hops_mag_synthetic_1.xlsx',
             'data/final_references_network_2hops_wos_synthetic_10.xlsx',
             'data/final_references_network_2hops_mag_synthetic_10.xlsx',
             'data/final_citations_network_2hops_mag_synthetic_10.xlsx',
             # 'data/final_best_features_synthetic_1.xlsx',
             # 'data/final_best_features_synthetic_10.xlsx'
             ]

    # mergeAllFeatures(files[:3], 'data/final_best_features_synthetic_1.xlsx')
    # mergeAllFeatures(files[3:], 'data/final_best_features_synthetic_10.xlsx')

    for i in range(len(files)):
        print('----------------Results for {} file----------------'.format(files[i].split('.')[0]))
        mscore = Model(all_labels[2], neural_model=False, fileName=files[i])
        mscore.get_data()
        features = mscore.select_best_features_chi2()
        b_features = list(features['Specs'])[: 10]
        mscore.modelling(b_features)
