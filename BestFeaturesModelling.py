from collections import defaultdict
import pandas as pd
from sklearn import ensemble
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras
import sklearn.metrics as mx
import copy


def select_best_features_chi2(df, label_type):
    X = df.drop([label_type, 'Authors.O'], axis=1)
    cols = X.columns
    X = MinMaxScaler().fit_transform(X)
    y = df[label_type]
    bestfeatures = SelectKBest(score_func=chi2, k='all')
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(cols)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    final_features = featureScores.nlargest(featureScores.shape[0] - 2, 'Score')
    pd.set_option('display.max_rows', final_features.shape[0] + 1)
    return final_features

def get_random_kfolds(df, train_set_size=0.8, seed=0, authors=None):
    if authors is None:
        return
    df = df.copy()
    train_set, test_set = pd.DataFrame(columns=list(df.columns)), pd.DataFrame(columns=list(df.columns))
    while len(train_set) <= df.shape[0] * train_set_size and len(authors) > 0 and df.shape[0] > 0:
        np.random.seed(seed)
        author = authors[np.random.randint(0, len(authors))]
        temp_df = df.loc[df['Authors.O'].str.contains(author, case=False)]
        train_set = pd.concat([train_set, temp_df], ignore_index=True)
        authors.remove(author)

        # remove all the rows from df that have been added to the new set
        df.drop(df[df['Authors.O'].str.contains(author, case=False)].index, inplace=True)

        if test_set.shape[0] <= df.shape[0] * (1 - train_set_size) and len(authors) > 0 and df.shape[0] > 0:
            np.random.seed(seed)
            author = authors[np.random.randint(0, len(authors))]
            temp_df = df.loc[df['Authors.O'].str.contains(author, case=False)]
            test_set = pd.concat([test_set, temp_df], ignore_index=True)
            authors.remove(author)

            # remove all the rows from df that have been added to the new set
            df.drop(df[df['Authors.O'].str.contains(author, case=False)].index, inplace=True)

    train_set = train_set.drop(['Authors.O'], axis=1)
    test_set = test_set.drop(['Authors.O'], axis=1)
    return train_set, test_set

def modelling_custom_kfolds(df, best_features, label_type, neural_model=True):
    gnb = GaussianNB()
    neigh = KNeighborsClassifier(n_neighbors=1, p=2, weights='uniform')
    forest = ensemble.RandomForestClassifier(random_state=0, n_estimators=5, max_depth=10, bootstrap=True)
    model = None
    if neural_model:
        model = keras.Sequential([
            keras.layers.Dense(32, input_dim=len(best_features), activation='sigmoid'),
            keras.layers.Dense(16, activation='sigmoid'),
            keras.layers.Dense(32, activation='sigmoid'),
            keras.layers.Dense(8, activation='sigmoid'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
    result = defaultdict(list)
    authors = []
    for au in list(df['Authors.O']):
        for j in au.split(','):
            author = j.strip().lower()
            if author not in authors:
                authors.append(author)
    for idx in range(10):
        print('Getting {} set '.format(idx + 1))
        train_set, test_set = get_random_kfolds(df, train_set_size=0.8, seed=idx, authors=copy.deepcopy(authors))
        X_train = train_set[best_features]
        y_train = train_set[label_type]
        y_train = y_train.astype('int')

        X_test = test_set[best_features]
        y_test = test_set[label_type]
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

        if neural_model:
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
            _, accuracy = model.evaluate(X_test, y_test)
            result['neuralNetwork'].append(accuracy)

    print("Accuracy of Naive Bayes is: %0.2f" % np.mean(result['gnb']))
    print("Accuracy of KNN is: %0.2f" % np.mean(result['neigh']))
    print("Accuracy of Random Forest is: %0.2f" % np.mean(result['forest']))
    if neural_model:
        print("Accuracy of Neural Network is: %0.2f" % np.mean(result['neuralNetwork']))

def get_baseline(df, label_type):
    total_true = total_false = total_val = 0
    for idx, row in df.iterrows():
        total_val += 1
        if row[label_type] == 1:
            total_true += 1
        else:
            total_false += 1
    print("Total rows is: ", total_val)
    print("Total papers reproducible: ", total_true)
    print("total_true % is: ", (total_true / total_val) * 100)
    print("total_false % is:", (total_false / total_val) * 100)

def tuning_hyperparameters(df, label_type, best_features):
    res_best = defaultdict(list)
    authors = []
    for au in list(df['Authors.O']):
        for j in au.split(','):
            author = j.strip().lower()
            if author not in authors:
                authors.append(author)
    for i in range(10):
        print('\n----------Round {}----------'.format(i + 1))
        train_set, test_set = get_random_kfolds(df, seed=i, train_set_size=0.8, authors=copy.deepcopy(authors))

        X_train = train_set[best_features]
        y_train = train_set[label_type]
        y_train = y_train.astype('int')

        X_test = test_set[best_features]
        y_test = test_set[label_type]
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
        for n_estimators in range(2, 11):
            for max_depth in range(2, 11):
                forest = ensemble.RandomForestClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth,
                                                         bootstrap=True)
                forest.fit(X_train, y_train)
                forest_pred = forest.predict(X_test)
                result[(n_estimators, max_depth)].append(np.round(mx.f1_score(y_test, forest_pred), 2))
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


if __name__ == '__main__':
    # df_ego = pd.read_csv('data/MAG_ego-paper-citation_measures.csv')
    # df_base = pd.read_excel('data/RPPdata.xlsx')
    #
    # for i, row in df_ego.iterrows():
    #     df_ego.at[i, 'Authors.O'] = df_base.loc[df_base['DOI'] == row['DOI']]['Authors.O'].values[0]
    #
    # df_ego.to_excel('data/MAG_ego_measures_with_Authors.xlsx')

    df_ego = pd.read_excel('data/MAG_ego_measures_with_Authors.xlsx')
    labels = ['Meta.analysis.significant', 'O.within.CI.R', 'P.value.R']

    df_ego.drop(['DOI', labels[0], labels[1], 'normalized_betweennesss_centrality', 'Unnamed: 0'], axis=1, inplace=True)
    df_ego.dropna(inplace=True)
    df_ego = df_ego.astype({'WOS_in-deg': 'float64', 'WOS_out-deg': 'float64', 'in-deg': 'float64', 'out-deg': 'float64',
                            'page_rank': 'float64', 'constraint': 'float64',
                            'betweennesss_centrality': 'float64', 'clustering_coefficient': 'float64'})
    print('After removing NAN rows the shape of the dataframe is: ', df_ego.shape)
    get_baseline(df_ego, labels[2])
    features = select_best_features_chi2(df_ego, labels[2])
    modelling_custom_kfolds(df_ego, list(features['Specs']), labels[2], neural_model=False)
    # tuning_hyperparameters(df_ego, labels[2], list(features['Specs']))

    # df_prev = pd.read_excel('data/final_bestfeatures_data.xlsx')
    # df_ego = pd.read_csv('data/MAG_ego-paper-citation_measures.csv')
    # for i, row in df_prev.iterrows():
    #     for prop in ['WOS_in-deg', 'WOS_out-deg', 'in-deg', 'out-deg', 'page_rank', 'betweennesss_centrality', 'normalized_betweennesss_centrality', 'clustering_coefficient', 'constraint']:
    #         df_prev.at[i, prop] = df_ego.loc[df_ego['DOI'] == row['DOI']][prop].values[0]
    #
    # df_prev.to_excel('data/final_bestfeatures_data_with_nazanins_features.xlsx')
