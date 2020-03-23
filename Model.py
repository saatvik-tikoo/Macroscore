from collections import defaultdict
import pandas
from sklearn import ensemble
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import numpy as np
from tensorflow import keras
import sklearn.metrics as mx
from matplotlib import pyplot as plt
import copy

class Model:
    def __init__(self, label_type, neural_model, fileName):
        self.label_type = label_type
        self.neural_model = neural_model
        self.fileName = fileName
        self.df = None

    def __score_model__(self, X_train, X_test, y_train, y_test, clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return mx.accuracy_score(y_test, y_pred)

    def ablation_test(self):
        self.__remove_unusable_features__()
        X = self.df.drop([self.label_type], axis=1)
        y = self.df[self.label_type]

        gnb = GaussianNB()
        svc = SVC(kernel='rbf', gamma=1, C=0.1, random_state=0)
        neigh = KNeighborsClassifier(n_neighbors=6, p=2, weights='uniform')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        base_score = dict()
        base_score['gnb'] = self.__score_model__(X_train, X_test, y_train, y_test, gnb)
        base_score['svc'] = self.__score_model__(X_train, X_test, y_train, y_test, svc)
        base_score['neigh'] = self.__score_model__(X_train, X_test, y_train, y_test, neigh)

        scores = defaultdict(list)
        for idx in range(X_train.shape[1]):
            cols = [ndx != idx for ndx in range(X_train.shape[1])]
            scores['gnb'].append(self.__score_model__(X_train.iloc[:, cols], X_test.iloc[:, cols], y_train, y_test, gnb))
            scores['svc'].append(self.__score_model__(X_train.iloc[:, cols], X_test.iloc[:, cols], y_train, y_test, svc))
            scores['neigh'].append(self.__score_model__(X_train.iloc[:, cols], X_test.iloc[:, cols], y_train, y_test, neigh))

        final_scores_gnb = dict()
        final_scores_svc = dict()
        final_scores_neigh = dict()
        for k, v in scores.items():
            if k == 'gnb':
                for idx in range(len(v)):
                    final_scores_gnb[X.columns[idx]] = (v[idx] - base_score[k])
                final_scores_gnb = sorted(final_scores_gnb.items(), key=lambda kv: kv[1], reverse=True)
            elif k == 'svc':
                for idx in range(len(v)):
                    final_scores_svc[X.columns[idx]] = (v[idx] - base_score[k])
                final_scores_svc = sorted(final_scores_svc.items(), key=lambda kv: kv[1], reverse=True)
            elif k == 'neigh':
                for idx in range(len(v)):
                    final_scores_neigh[X.columns[idx]] = (v[idx] - base_score[k])
                final_scores_neigh = sorted(final_scores_neigh.items(), key=lambda kv: kv[1], reverse=True)

        print('Based on Naive Bayes: ', final_scores_gnb)
        print('Based on SVC : ', final_scores_svc)
        print('Based on KNN: ', final_scores_neigh)

    def __remove_unusable_features__(self):
        print('Initial Shape is: ', self.df.shape)
        if self.label_type == 'pvalue.label':
            self.df = self.df.drop(['P.value.R', 'Direction.R', 'O.within.CI.R', 'Meta.analysis.significant'], axis=1)
        elif self.label_type == 'O.within.CI.R':
            self.df = self.df.drop(['P.value.R', 'Direction.R', 'Meta.analysis.significant', 'pvalue.label'], axis=1)
        elif self.label_type == 'Meta.analysis.significant':
            self.df = self.df.drop(['P.value.R', 'Direction.R', 'O.within.CI.R', 'pvalue.label'], axis=1)

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

    def modelling(self, best_features):
        self.df = self.df.drop(['Authors.O'], axis=1)
        X = self.df[best_features]
        y = self.df[self.label_type]
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        gnb = GaussianNB()
        svc = SVC(kernel='rbf', gamma=0.1, C=0.1, random_state=0)
        neigh = KNeighborsClassifier(n_neighbors=6, p=3, weights='uniform')
        forest = ensemble.RandomForestClassifier(random_state=0, n_estimators=5, max_features='auto', max_depth=10,
                                                 min_samples_split=2, min_samples_leaf=1, bootstrap=True)
        print("Cross Validation Score of Naive Bayes is: %.2f" % np.mean(cross_val_score(gnb, X, y, cv=skf, n_jobs=1)))
        print("Cross Validation Score of SVC is: %.2f" % np.mean(cross_val_score(svc, X, y, cv=skf, n_jobs=1)))
        print("Cross Validation Score of KNN is: %.2f" % np.mean(cross_val_score(neigh, X, y, cv=skf, n_jobs=1)))
        print("Cross Validation Score of Random Forest is: %.2f" % np.mean(cross_val_score(forest, X, y, cv=skf, n_jobs=1)))

        if self.neural_model:
            acc_arr = []
            model = keras.Sequential([
                keras.layers.Dense(16, input_dim=X.shape[1], activation='sigmoid'),
                keras.layers.Dense(8, activation='sigmoid'),
                keras.layers.Dense(16, activation='sigmoid'),
                keras.layers.Dense(8, activation='sigmoid'),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
                _, accuracy = model.evaluate(X_test, y_test)
                acc_arr.append(accuracy)
            print('Accuracy of this neural network model is: %.2f' % np.mean(acc_arr))
            print(acc_arr)

    def modelling_custom_kfolds(self, best_features):
        gnb = GaussianNB()
        svc = SVC(kernel='rbf', gamma=0.1, C=0.1, random_state=0)
        neigh = KNeighborsClassifier(n_neighbors=12, p=3, weights='uniform')
        forest = ensemble.RandomForestClassifier(random_state=0, n_estimators=7, max_depth=4, bootstrap=True)
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
        authors = []
        for au in list(self.df['Authors.O']):
            for j in au.split(','):
                author = j.strip().lower()
                if author not in authors:
                    authors.append(author)
        for idx in range(10):
            print('Getting {} set '.format(idx + 1))
            train_set, test_set = self.get_random_kfolds(train_set_size=0.8, seed=idx, authors=copy.deepcopy(authors))
            X_train = train_set[best_features]
            y_train = train_set[self.label_type]
            y_train = y_train.astype('int')

            X_test = test_set[best_features]
            y_test = test_set[self.label_type]
            y_test = y_test.astype('int')

            gnb.fit(X_train, y_train)
            gnb_pred = gnb.predict(X_test)
            result['gnb'].append(mx.accuracy_score(y_test, gnb_pred))

            svc.fit(X_train, y_train)
            svc_pred = svc.predict(X_test)
            result['svc'].append(mx.accuracy_score(y_test, svc_pred))

            neigh.fit(X_train, y_train)
            neigh_pred = neigh.predict(X_test)
            result['neigh'].append(mx.accuracy_score(y_test, neigh_pred))

            forest.fit(X_train, y_train)
            forest_pred = forest.predict(X_test)
            result['forest'].append(mx.accuracy_score(y_test, forest_pred))

            if self.neural_model:
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
                _, accuracy = model.evaluate(X_test, y_test)
                result['neuralNetwork'].append(accuracy)

        print("Accuracy of Naive Bayes is: %0.2f" % np.mean(result['gnb']))
        print("Accuracy of SVC is: %0.2f" % np.mean(result['svc']))
        print("Accuracy of KNN is: %0.2f" % np.mean(result['neigh']))
        print("Accuracy of Random Forest is: %0.2f" % np.mean(result['forest']))
        if self.neural_model:
            print("Accuracy of Neural Network is: %0.2f" % np.mean(result['neuralNetwork']))

    def tuning_hyperparameters(self, best_features):
        res_best = defaultdict(list)
        authors = []
        for au in list(self.df['Authors.O']):
            for j in au.split(','):
                author = j.strip().lower()
                if author not in authors:
                    authors.append(author)
        for idx in range(10):
            print('\n----------Round {}----------'.format(idx + 1))
            train_set, test_set = self.get_random_kfolds(train_set_size=0.8, seed=idx, authors=copy.deepcopy(authors))

            X_train = train_set[best_features]
            y_train = train_set[self.label_type]
            y_train = y_train.astype('int')

            X_test = test_set[best_features]
            y_test = test_set[self.label_type]
            y_test = y_test.astype('int')

            result = defaultdict(list)
            gamma_val = c_val = 0.01
            while gamma_val < 10:
                while c_val < 10:
                    svc = SVC(kernel='rbf', gamma=gamma_val, C=c_val, random_state=0)
                    svc.fit(X_train, y_train)
                    svc_pred = svc.predict(X_test)
                    result[(gamma_val, c_val)].append(np.round(mx.accuracy_score(y_test, svc_pred), 2))
                    c_val += 0.01
                gamma_val += 0.01
            final_val = max(result.items(), key=lambda x: x[1])
            print("Accuracy of SVC is: {}".format(final_val))
            res_best['svc'].append(final_val)

            result = defaultdict(list)
            for n_neighbors_val in range(3, 15):
                for p_val in range(1, 6):
                    neigh = KNeighborsClassifier(n_neighbors=n_neighbors_val, p=p_val, weights='uniform')
                    neigh.fit(X_train, y_train)
                    neigh_pred = neigh.predict(X_test)
                    result[(n_neighbors_val, p_val)].append(np.round(mx.accuracy_score(y_test, neigh_pred), 2))
            final_val = max(result.items(), key=lambda x: x[1])
            print("Accuracy of KNN is: {}".format(final_val))
            res_best['knn'].append(final_val)

            result = defaultdict(list)
            for n_estimators in range(2, 11):
                for max_depth in range(2, 11):
                    forest = ensemble.RandomForestClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth,
                                                             bootstrap=True)
                    forest.fit(X_train, y_train)
                    forest_pred = forest.predict(X_test)
                    result[(n_estimators, max_depth)].append(np.round(mx.accuracy_score(y_test, forest_pred), 2))
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


if __name__ == '__main__':
    all_labels = ['pvalue.label', 'O.within.CI.R', 'Meta.analysis.significant']
    files = ['data/final_references_wos_data.xlsx', 'data/final_references_mag_data.xlsx',
             'data/final_citations_mag_data.xlsx', 'data/final_bestfeatures_data.xlsx']
    # full_df = pandas.DataFrame()
    for i in range(len(files)):
        print('----------------Results for {} file----------------'.format(files[i].split('.')[0]))
        mscore = Model(all_labels[0], neural_model=False, fileName=files[i])
        mscore.get_data()
        features = mscore.select_best_features_chi2()
        b_features = list(features['Specs'])[:10]
        if i < len(files) - 1:
            # mscore.modelling(b_features)
            mscore.modelling_custom_kfolds(b_features)
        else:
            # mscore.modelling(list(features['Specs']))
            mscore.modelling_custom_kfolds(list(features['Specs']))

        # mscore.tuning_hyperparameters(b_features)

        # df_temp = pandas.read_excel(files[i], encoding='ansi')
        # if i == 0:
        #     full_df = pandas.concat([full_df, df_temp[['DOI', 'P.value.R', 'Direction.R', 'O.within.CI.R',
        #                                                'Meta.analysis.significant', 'pvalue.label',
        #                                                'Authors.O']]], axis=1)
        #
        # full_df = pandas.concat([full_df, df_temp[b_features]], axis=1)
    # full_df.to_excel('data/final_bestfeatures_data.xlsx')

