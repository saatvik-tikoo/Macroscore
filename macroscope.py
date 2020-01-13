import pandas
import csv
import pandas as pd
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import auc, roc_curve, accuracy_score
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

import tensorflow as tf
from tensorflow import keras
import numpy as np

def label_addition(row):
    try:
        if 'X' in str(row['P-value (R)']) or str(row['P-value (R)']) == 'nan' or 'significant' in str(row['P-value (R)']):
            return False
        if '<' in str(row['P-value (R)']):
            val = "{0:.4f}".format(float(row['P-value (R)'].split('<')[1].strip()) - float(row['P-value (R)'].split('<')[1].strip()) / 10)
        elif '>' in str(row['P-value (R)']):
            val = "{0:.4f}".format(float(row['P-value (R)'].split('>')[1].strip()) + float(row['P-value (R)'].split('>')[1].strip()) / 10)
        elif '=' in str(row['P-value (R)']):
            val = row['P-value (R)'].split('=')[1].strip()
        elif 'not significant' in str(row['P-value (R)']) and row['Direction (R)'] == 'same':
            return True
        elif float(row['P-value (R)']) <= 0.05 and row['Direction (R)'] == 'same':
            return True
        else:
            return False

        if float(val) <= 0.05 and row['Direction (R)'] == 'same':
            return True
        else:
            return False
    except ValueError:
        return False

def clean_pvalue(data_set, prop):
    for i, row in data_set.iterrows():
        if 'significant' in str(row[prop]):
            data_set.at[i, prop] = '1'
        if '<' in str(row[prop]):
            data_set.at[i, prop] = "{0:.4f}".format(float(row[prop].split('<')[1].strip()) - float(row[prop].split('<')[1].strip()) / 2)
        elif '>' in str(row[prop]):
            data_set.at[i, prop] = "{0:.4f}".format(float(row[prop].split('>')[1].strip()) + float(row[prop].split('>')[1].strip()) / 2)
        elif '=' in str(row[prop]):
            data_set.at[i, prop] = row[prop].split('=')[1].strip()
        elif 'not significant' in str(row[prop]):
            data_set.at[i, prop] = '0'
    return data_set

def clean_float_result(data_set, prop):
    for i, row in data_set.iterrows():
        try:
            if float(row[prop]):
                continue
        except ValueError:
            data_set.at[i, prop] = 0
    return data_set

def clean_range_values(data_set, prop):
    for i, row in data_set.iterrows():
        temp = []
        if '-' in str(row[prop]):
            temp = str(row[prop]).split('-')
            data_set.at[i, prop] = abs(float(temp[0]) - float(temp[0])) + 1
    return data_set

def clean_data():
    # Get required fields of the data-set (fields not of the Replication Study)
    req_columns = ['P-value (R)', 'Direction (R)']
    remove_columns = ['Study Title (O)', 'Descriptors (O)', 'Description of effect (O)', 'Feasibility (O)',
                      'Calculated P-value (O)', 'Test statistic (O)', 'Effect size (O)', 'Type of analysis (O)']
    data_types = ['Integer', 'Range', 'Categorical', 'Decimal', 'Dichotomous', 'Open text']
    with open("Psychology/rpp_data_codebook.csv") as input_file:
        reader = csv.reader(input_file)
        # skip the header
        next(reader)
        for line in reader:
            if '(R)' not in line[0] and line[4] in data_types and line[0] not in remove_columns:
                req_columns.append(line[0])
    data_set = pandas.read_csv('Psychology/rpp_data_updates.csv', encoding='ansi', usecols=req_columns)

    # Add output field and dropping the Replication fields
    data_set['result_label'] = data_set.apply(lambda row: label_addition(row), axis=1)
    data_set = data_set.drop(['P-value (R)', 'Direction (R)'], axis=1)

    # Clean fields
    data_set = data_set.replace(to_replace='X', value=np.nan)
    data_set = data_set.replace(to_replace=np.nan, value=0)
    data_set = clean_pvalue(data_set, 'Reported P-value (O)')
    data_set = clean_range_values(data_set, 'Pages (O)')
    data_set = clean_float_result(data_set, 'Surprising result (O)')
    data_set = clean_float_result(data_set, 'Exciting result (O)')

    # Removing the last row as it is a null row
    data_set = data_set[: -1]

    # Changing the data-types of some fields so as not to include them in encoding
    data_set = data_set.astype({'Reported P-value (O)': 'float64', 'Pages (O)': 'float64', 'Surprising result (O)': 'float64',
                                'Exciting result (O)': 'float64', 'N (O)': 'float64', '# Tails (O)': 'float64'})

    # Do a separate encoding for authors
    all_authors = data_set['Authors (O)'].str.split(',', expand=True).stack()
    temp = pd.get_dummies(all_authors, prefix='g').groupby(level=0).sum()
    data_set = data_set.drop(['Authors (O)'], axis=1)
    # temp.to_excel('Psychology/Authors.xlsx')
    print(temp.shape)

    # Add the encoded authors to the rest of data
    # data_set.to_excel('Psychology/RawData.xlsx')
    data_set = pd.get_dummies(data_set, drop_first=True)
    data_set = pd.concat([temp, data_set], axis=1)
    # data_set.to_excel('Psychology/JustCreatedFile.xlsx')
    print(data_set.shape)

    total_true = total_false = total_val = 0
    for i, row in data_set.iterrows():
        total_val += 1
        if row['result_label']:
            total_true += 1
        else:
            total_false += 1
    print("total_true % is= ", (total_true / total_val) * 100)
    print("total_false % is=", (total_false / total_val) * 100)
    return data_set

def modelling(df, cross_validation=None):
    X = df.drop(['result_label'], axis=1)
    y = df['result_label']
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    gnb = GaussianNB()
    Svc = SVC(random_state=0)
    LSvc = LinearSVC(random_state=0, C=0.01, dual=False)
    neigh = KNeighborsClassifier(n_neighbors=4, p=2)
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_features='auto',
                                      min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                                      random_state=0, splitter='best')
    print("Cross Validation Score of NB is: ", np.mean(cross_val_score(gnb, X, y, cv=skf, n_jobs=1)))
    print("Cross Validation Score of SVC is: ", np.mean(cross_val_score(Svc, X, y, cv=skf, n_jobs=1)))
    print("Cross Validation Score of Linear SVC is: ", np.mean(cross_val_score(LSvc, X, y, cv=skf, n_jobs=1)))
    print("Cross Validation Score of KNN is: ", np.mean(cross_val_score(neigh, X, y, cv=skf, n_jobs=1)))
    print("Cross Validation Score of Decision Tree is: ", np.mean(cross_val_score(clf, X, y, cv=skf, n_jobs=1)))

    acc_arr = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = keras.Sequential([
            keras.layers.Dense(128, input_dim=915, activation='sigmoid'),
            keras.layers.Dense(256, activation='sigmoid'),
            keras.layers.Dense(128, activation='sigmoid'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
        _, accuracy = model.evaluate(X_test, y_test)
        acc_arr.append(accuracy)
    print('Accuracy of this model is: %.2f' % (np.mean(acc_arr) * 100))
    print(acc_arr)

def clean_data_dict_vectorize():
    # Get required fields of the data-set (fields not of the Replication Study)
    req_columns = ['P-value (R)', 'Direction (R)']
    remove_columns = ['Study Title (O)', 'Descriptors (O)', 'Description of effect (O)',
                      'Calculated P-value (O)', 'Test statistic (O)', 'Effect size (O)', 'Type of analysis (O)']
    data_types = ['Integer', 'Range', 'Categorical', 'Decimal', 'Dichotomous', 'Open text']
    with open("Psychology/rpp_data_codebook.csv") as input_file:
        reader = csv.reader(input_file)
        # skip the header
        next(reader)
        for line in reader:
            if '(R)' not in line[0] and line[4] in data_types and line[0] not in remove_columns:
                req_columns.append(line[0])
    data_set = pandas.read_csv('Psychology/rpp_data_updates.csv', encoding='ansi', usecols=req_columns)

    # Add output field
    data_set['result_label'] = data_set.apply(lambda row: label_addition(row), axis=1)
    data_set = data_set.drop(['P-value (R)', 'Direction (R)'], axis=1)

    # Clean fields: Reported P-value
    data_set = data_set.replace(to_replace='X', value=np.nan)
    data_set = data_set.replace(to_replace=np.nan, value=0)
    data_set = clean_pvalue(data_set, 'Reported P-value (O)')
    data_set = clean_range_values(data_set, 'Pages (O)')
    data_set = clean_float_result(data_set, 'Surprising result (O)')
    data_set = clean_float_result(data_set, 'Exciting result (O)')
    data_set = data_set[: -1]
    data_set = data_set.astype({'Reported P-value (O)': 'float64', 'Pages (O)': 'float64', 'Surprising result (O)': 'float64',
                                'Exciting result (O)': 'float64', 'N (O)': 'float64', '# Tails (O)': 'float64'})
    all_authors = data_set['Authors (O)'].str.split(',', expand=True).stack()
    temp = pd.get_dummies(all_authors, prefix='g').groupby(level=0).sum()
    data_set = data_set.drop(['Authors (O)'], axis=1)
    data_set = pd.concat([temp, data_set], axis=1)

    X = data_set.drop(['result_label'], axis=1)
    y = data_set['result_label']

    X_dict = X.to_dict(orient='records')
    dv_X = DictVectorizer(sparse=False)
    X = dv_X.fit_transform(X_dict)
    # data_set.to_excel('Psychology/JustCreatedFile.xlsx')
    print(X.shape)
    return X, y

def modelling_dictvectorize(X_income, y_income):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    gnb = GaussianNB()
    Svc = SVC(random_state=0)
    LSvc = LinearSVC(random_state=0, C=0.01, dual=False)
    neigh = KNeighborsClassifier(n_neighbors=4, p=2)
    print("Cross Validation Score of NB using DictVector is: ", np.mean(cross_val_score(gnb, X_income, y_income, cv=skf, n_jobs=1)))
    print("Cross Validation Score of SVC using DictVector is: ", np.mean(cross_val_score(Svc, X_income, y_income, cv=skf, n_jobs=1)))
    print("Cross Validation Score of Linear SVC using DictVector is: ", np.mean(cross_val_score(LSvc, X_income, y_income, cv=skf, n_jobs=1)))
    print("Cross Validation Score of KNN using DictVector is: ", np.mean(cross_val_score(neigh, X_income, y_income, cv=skf, n_jobs=1)))
    acc_arr = []
    for train_index, test_index in skf.split(X_income, y_income):
        X_train, X_test = X_income[train_index], X_income[test_index]
        y_train, y_test = y_income[train_index], y_income[test_index]
        model = keras.Sequential([
            keras.layers.Dense(128, input_dim=927, activation='sigmoid'),
            keras.layers.Dense(256, activation='sigmoid'),
            keras.layers.Dense(128, activation='sigmoid'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
        _, accuracy = model.evaluate(X_test, y_test)
        acc_arr.append(accuracy)
    print('Accuracy of this model is: %.2f' % (np.mean(acc_arr) * 100))
    print(acc_arr)


if __name__ == '__main__':
    # one hot encoding
    df = clean_data()
    modelling(df)
    # dictVectorize encoding
    # X, y = clean_data_dict_vectorize()
    # modelling_dictvectorize(X, y)
