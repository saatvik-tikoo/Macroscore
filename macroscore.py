import os
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
import xgboost as xgb


class Macroscore:
    def __init__(self, label_type, feature_type='all', encoding='oneHot', specify_features=False,
                 features=None, neural_model=False, fileName=''):
        # label_type: {'pvalue.label', 'O.within.CI.R', 'Meta.analysis.significant'}
        self.label_type = label_type
        # features: {'all', 'common', 'network'}
        self.feature_type = feature_type
        # encoding: {'oneHot', 'label', 'DictVectroize'}
        self.encoding = encoding
        # if only specific features are required.
        self.specify_features = specify_features
        # list of features if specify_features is true
        self.features = features
        # true if neural network model is required
        self.neural_model = neural_model
        self.fileName = fileName
        self.df = None
        self.path_head = '../DataExtraction/WOS/RPPdataConverted'

    def __label_addition__(self, row):
        try:
            if 'X' in str(row['P.value.R']) or str(row['P.value.R']) == 'nan' or 'significant' in str(row['P.value.R']):
                return 0
            if '<' in str(row['P.value.R']):
                val = "{0:.4f}".format(float(row['P.value.R'].split('<')[1].strip()) - float(row['P.value.R'].split('<')[1].strip()) / 10)
            elif '>' in str(row['P.value.R']):
                val = "{0:.4f}".format(float(row['P.value.R'].split('>')[1].strip()) + float(row['P.value.R'].split('>')[1].strip()) / 10)
            elif '=' in str(row['P.value.R']):
                val = row['P.value.R'].split('=')[1].strip()
            elif 'not significant' in str(row['P.value.R']) and row['Direction.R'] == 'same':
                return 1
            elif float(row['P.value.R']) <= 0.05 and row['Direction.R'] == 'same':
                return 1
            else:
                return 0

            if float(val) <= 0.05 and row['Direction.R'] == 'same':
                return 1
            else:
                return 0
        except ValueError:
            return 0

    def __clean_pvalue__(self, prop):
        for i, row in self.df.iterrows():
            if 'significant' in str(row[prop]):
                self.df.at[i, prop] = '1'
            if '<' in str(row[prop]):
                self.df.at[i, prop] = "{0:.4f}".format(float(row[prop].split('<')[1].strip()) - float(row[prop].split('<')[1].strip()) / 2)
            elif '>' in str(row[prop]):
                self.df.at[i, prop] = "{0:.4f}".format(float(row[prop].split('>')[1].strip()) + float(row[prop].split('>')[1].strip()) / 2)
            elif '=' in str(row[prop]):
                self.df.at[i, prop] = row[prop].split('=')[1].strip()
            elif 'not significant' in str(row[prop]):
                self.df.at[i, prop] = '0'

    def __clean_float_result__(self, prop):
        for i, row in self.df.iterrows():
            try:
                if float(row[prop]):
                    continue
            except ValueError:
                self.df.at[i, prop] = 0

    def __clean_effect_size__(self, prop):
        for i, row in self.df.iterrows():
            if '=' in str(row[prop]):
                self.df.at[i, prop] = float(str(row[prop]).split('=')[1].strip())

    def __get_baseline__(self):
        total_true = total_false = total_val = 0
        for i, row in self.df.iterrows():
            total_val += 1
            if row[self.label_type] == 1:
                total_true += 1
            else:
                total_false += 1
        print("Total rows is= ", total_val)
        print("Total papers reproducible", total_true)
        print("total_true % is= ", (total_true / total_val) * 100)
        print("total_false % is=", (total_false / total_val) * 100)

    def __clean_data__(self):
        self.df = self.df.replace(to_replace='X', value=np.nan)
        self.df = self.df.dropna(subset=['DOI', 'Reported.P.value.O', 'Direction.R', 'Meta.analysis.significant', 'O.within.CI.R'])
        self.df = self.df.replace(to_replace=np.nan, value=0)
        self.__clean_pvalue__('Reported.P.value.O')
        self.__clean_float_result__('Surprising.result.O')
        self.__clean_float_result__('Exciting.result.O')
        self.__clean_effect_size__('Effect.size.O')

        if self.label_type == 'pvalue.label':
            self.df['pvalue.label'] = self.df.apply(lambda paper: self.__label_addition__(paper), axis=1)

    def __remove_unusable_features__(self):
        if self.label_type == 'pvalue.label':
            self.df = self.df.drop(['P.value.R', 'Direction.R', 'O.within.CI.R', 'Meta.analysis.significant'], axis=1)
        elif self.label_type == 'O.within.CI.R':
            self.df = self.df.drop(['P.value.R', 'Direction.R', 'Meta.analysis.significant', 'pvalue.label'], axis=1)
        elif self.label_type == 'Meta.analysis.significant':
            self.df = self.df.drop(['P.value.R', 'Direction.R', 'O.within.CI.R', 'pvalue.label'], axis=1)

        cols_drop = set(['DOI', '1st.author.O', 'Senior.author.O', 'Study.Title.O', 'Unnamed: 0', 'new_feature_301', 'Unnamed: 0.1'])
        cols_total = set(self.df.columns)
        self.df = self.df.drop(cols_drop.intersection(cols_total), axis=1)
        # self.df = self.df.replace(to_replace=np.nan, value=0)
        self.df = self.df.dropna()
        print('Shape is: ', self.df.shape)

    def __get__common_features__(self):
        self.__clean_data__()

        self.__get_baseline__()

        # Changing the data-types of some fields so as not to include them in encoding
        self.df = self.df.astype({'Reported.P.value.O': 'float64', 'Exciting.result.O': 'float64', 'Surprising.result.O': 'float64', 'N.O': 'float64',
                                  'Effect.size.O': 'float64'})
        return self.df

    def __get_network_features__(self):
        print('Initial Shape is: ', self.df.shape)
        if self.feature_type == 'network':
            self.__clean_data__()
            self.__get_baseline__()
        for i, row in self.df.iterrows():
            name = row['DOI'].replace('/', '_')
            print('----- Getting details for ', row['DOI'], '-------')

            # Get Citations related data
            file_name = self.path_head + '/{}/paper_{}.txt'.format(name, name)
            all_authors_paper = []
            if os.path.exists(file_name):
                df_doi = pandas.read_csv(file_name, sep='\t', lineterminator='\r', encoding="utf-16le", index_col=False,
                                         quotechar=None, quoting=3, usecols=['AU', 'NR', 'TC'])
                df_doi = df_doi.dropna()
                all_authors_paper = df_doi['AU'].str.split(';')[0]
                self.df.at[i, 'References'] = df_doi['NR'][0]
                self.df.at[i, 'Citation.count.paper.O'] = df_doi['TC'][0]
                print('Total references', df_doi['NR'][0])
                print('Total Citations', df_doi['TC'][0])

            # Get References data: References to same authors
            file_name = self.path_head + '/{}/citations_{}.txt'.format(name, name)
            if os.path.exists(file_name):
                df_doi = pandas.read_csv(file_name, sep='\t', lineterminator='\r', encoding="utf-16le", index_col=False,
                                         quotechar=None, quoting=3, usecols=['AU'])
                df_doi = df_doi.dropna()
                cnt_authors = 0
                for _, row_internal in df_doi.iterrows():
                    authors_ref = row_internal['AU'].split(';')
                    for j in authors_ref:
                        if j in all_authors_paper:
                            cnt_authors += 1
                self.df.at[i, 'References.to.self'] = cnt_authors
                print('Where they have referred themselves: ', cnt_authors)

            # Get Authors data
            for author_col in ['1st.author.O', 'Senior.author.O']:
                folder_name = self.path_head + '/{}/'.format(name)
                keyword = row[author_col]
                if os.path.exists(folder_name):
                    for file in os.listdir(folder_name):
                        if keyword in file:
                            df_doi = pandas.read_csv(folder_name + '/' + file, sep='\t', lineterminator='\r', encoding="utf-16le", index_col=False,
                                                     quotechar=None, quoting=3, usecols=['TC'])
                            df_doi = df_doi.dropna()
                            self.df.at[i, author_col + ' papers'] = cnt_authors
                            print('Number of papers of ' + author_col + ' :', df_doi.shape[0])
                            res = 0
                            for _, row_internal in df_doi.iterrows():
                                res += row_internal['TC']
                            self.df.at[i, author_col + ' citations.of.all.papers'] = res
                            print('Number of Citations of  ' + author_col + ' :', res)
                            break

    def get_data(self):
        if not self.specify_features:
            self.df = pandas.read_excel(self.fileName, encoding='ansi')
        else:
            self.df = pandas.read_excel(self.fileName, encoding='ansi', usecols=self.features)

    def get_feature(self):
        if self.feature_type.lower() == 'common':
            self.__get__common_features__()
        elif self.feature_type.lower() == 'network':
            self.__get_network_features__()
        elif self.feature_type.lower() == 'all':
            self.__get__common_features__()
            self.__get_network_features__()
        else:
            print('Wrong features asked: ', self.features)
            return
        print('Final Shape is: ', self.df.shape)
        self.df.to_excel('data/new_data.xlsx')
        self.__get_baseline__()

    # Just call this function 10 times
    def get_random_kfolds(self, k):
        authors = set([j.strip() for i in list(self.df['Authors.O']) for j in i.split(',')])
        test_set, train_set = pandas.DataFrame(columns=list(self.df.columns)), pandas.DataFrame(columns=list(self.df.columns))
        while len(test_set) <= self.df.shape[0] * (k / 100):
            author = authors[np.random.randint(0, len(author))]
            pandas.concat([test_set, self.df.where(author in self.df['Authors.O'])])

            # remove all the authors from the list that have been added here
            test_set_authors = set([j.strip() for i in list(test_set['Authors.O']) for j in i.split(',')])
            authors = authors - test_set_authors

            if train_set.shape[0] <= self.df.shape[0] * ((k - 1) / 100):
                author = authors[np.random.randint(0, len(author))]
                pandas.concat([train_set, self.df.where(author in self.df['Authors.O'])])

                # remove all the authors from the list that have been added here
                train_set_authors = set([j.strip() for i in list(test_set['Authors.O']) for j in i.split(',')])
                authors = authors - train_set_authors
        return test_set, train_set

    def modelling(self):
        self.__remove_unusable_features__()
        self.__get_baseline__()
        # X = self.df.drop([self.label_type], axis=1)
        X = self.df[['new_feature_63', 'new_feature_93', 'new_feature_78', 'new_feature_139', 'new_feature_1',
                     'new_feature_75', 'new_feature_292', 'new_feature_42', 'new_feature_111', 'new_feature_183']]
        y = self.df[self.label_type]
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        gnb = GaussianNB()
        Svc = SVC(kernel='rbf', gamma=0.9, C=1, random_state=0)
        neigh = KNeighborsClassifier(n_neighbors=5, p=2, weights='uniform')
        forest = ensemble.RandomForestClassifier(random_state=0, n_estimators=10, max_features='auto', max_depth=10,
                                                 min_samples_split=2, min_samples_leaf=1, bootstrap=True)
        print("Cross Validation Score of NB is: ", np.mean(cross_val_score(gnb, X, y, cv=skf, n_jobs=1)))
        print("Cross Validation Score of SVC is: ", np.mean(cross_val_score(Svc, X, y, cv=skf, n_jobs=1)))
        print("Cross Validation Score of KNN is: ", np.mean(cross_val_score(neigh, X, y, cv=skf, n_jobs=1)))
        print("Cross Validation Score of Random Forest is: ", np.mean(cross_val_score(forest, X, y, cv=skf, n_jobs=1)))

        xgboost = xgb.XGBClassifier(max_depth=6, objective='binary:logistic', learning_rate=1, colsample_bytree=1, reg_alpha=5, booster='gbtree')
        print("Cross Validation Score of XGB is: ", np.mean(cross_val_score(xgboost, X, y, cv=skf, n_jobs=1)))

        if self.neural_model:
            acc_arr = []
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model = keras.Sequential([
                    keras.layers.Dense(16, input_dim=X.shape[1], activation='sigmoid'),
                    keras.layers.Dense(8, activation='sigmoid'),
                    keras.layers.Dense(16, activation='sigmoid'),
                    keras.layers.Dense(8, activation='sigmoid'),
                    keras.layers.Dense(1, activation='sigmoid')
                ])
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
                _, accuracy = model.evaluate(X_test, y_test)
                acc_arr.append(accuracy)
            print('Accuracy of this neural network model is: %.2f' % (np.mean(acc_arr) * 100))
            print(acc_arr)

    def select_best_features_chi2(self):
        self.df = pandas.read_excel('data/final_network_data.xlsx', encoding='ansi')
        self.__remove_unusable_features__()
        X = self.df.drop([self.label_type], axis=1)
        cols = X.columns
        X = MinMaxScaler().fit_transform(X)
        y = self.df[self.label_type]
        # apply SelectKBest class to extract top 10 best features
        bestfeatures = SelectKBest(score_func=chi2, k=10)
        fit = bestfeatures.fit(X, y)
        dfscores = pandas.DataFrame(fit.scores_)
        dfcolumns = pandas.DataFrame(cols)
        featureScores = pandas.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']
        final_features = featureScores.nlargest(featureScores.shape[0] - 2, 'Score')
        pandas.set_option('display.max_rows', final_features.shape[0] + 1)
        return final_features

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
        for i in range(X_train.shape[1]):
            cols = [ndx != i for ndx in range(X_train.shape[1])]
            scores['gnb'].append(self.__score_model__(X_train.iloc[:, cols], X_test.iloc[:, cols], y_train, y_test, gnb))
            scores['svc'].append(self.__score_model__(X_train.iloc[:, cols], X_test.iloc[:, cols], y_train, y_test, svc))
            scores['neigh'].append(self.__score_model__(X_train.iloc[:, cols], X_test.iloc[:, cols], y_train, y_test, neigh))

        final_scores_gnb = dict()
        final_scores_svc = dict()
        final_scores_neigh = dict()
        for k, v in scores.items():
            if k == 'gnb':
                for i in range(len(v)):
                    final_scores_gnb[X.columns[i]] = (v[i] - base_score[k])
                final_scores_gnb = sorted(final_scores_gnb.items(), key=lambda kv: kv[1], reverse=True)
            elif k == 'svc':
                for i in range(len(v)):
                    final_scores_svc[X.columns[i]] = (v[i] - base_score[k])
                final_scores_svc = sorted(final_scores_svc.items(), key=lambda kv: kv[1], reverse=True)
            elif k == 'neigh':
                for i in range(len(v)):
                    final_scores_neigh[X.columns[i]] = (v[i] - base_score[k])
                final_scores_neigh = sorted(final_scores_neigh.items(), key=lambda kv: kv[1], reverse=True)

        print('Based on NB: ', final_scores_gnb)
        print('Based on SVC : ', final_scores_svc)
        print('Based on KNN: ', final_scores_neigh)

    def plot_feature_graph(self, cols):
        i = 10
        sctr_plot = dict()
        self.__get_baseline__()
        y = self.df[self.label_type]
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        while i < cols.shape[0]:
            lst = cols['Specs'].head(i)
            X = self.df[lst]
            neigh = KNeighborsClassifier(n_neighbors=5, p=2, weights='uniform')
            sctr_plot[i] = np.mean(cross_val_score(neigh, X, y, cv=skf, n_jobs=1))
            i += 1
        x = sctr_plot.keys()
        y = sctr_plot.values()
        plt.scatter(x, y)
        plt.show()


if __name__ == '__main__':
    req_columns = ['Study.Title.O', 'Authors.O', 'Volume.O', 'Citation.count.paper.O', 'Number.of.Authors.O', 'DOI', 'Citation.Count.1st.author.O',
                   'Reported.P.value.O', 'Exciting.result.O', 'Surprising.result.O', 'N.O', 'Effect.size.O',
                   'Institution.prestige.1st.author.O', 'Institution.prestige.senior.author.O', 'O.within.CI.R', 'P.value.R',
                   'Direction.R', 'Meta.analysis.significant', 'Citation.count.senior.author.O', '1st.author.O',
                   'Senior.author.O']

    # Uncomment below to clean data and add network features: Step-1
    mscore = Macroscore('pvalue.label', feature_type='all', specify_features=True, features=req_columns,
                        neural_model=True, fileName='data/RPPdata.xlsx')
    mscore.get_data()
    mscore.get_feature()

    # Uncomment below to train and test our models: Step-5
    # mscore = Macroscore('pvalue.label', feature_type='all', specify_features=False,
    #                     neural_model=True, fileName='data/final_network_data.xlsx')
    # mscore.get_data()
    # mscore.modelling()

    # Various Tests
    # features = mscore.select_best_features_chi2()
    # print(features)
    # mscore.plot_feature_graph(features)
    # mscore.ablation_test()

