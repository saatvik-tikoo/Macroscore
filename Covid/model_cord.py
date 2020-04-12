import pickle
import pandas as pd
from random import randint, seed


if __name__ == '__main__':
    print('-----Reading Files-----')
    f = open('data/models.pkl', 'rb')
    gnb = pickle.load(f)
    neigh = pickle.load(f)
    forest = pickle.load(f)

    files = ['finaldata_whole']

    for i in files:
        df = pd.read_excel('data/covid19_kb/{}.xlsx'.format(i))
        print('-----Getting Random Numbers-----')
        seed(0)
        df_subset = df.drop(['doi', 'Unnamed: 0'], axis=1).copy()
        # df_subset = df_subset.drop(df_subset.columns[randint(0, df_subset.shape[1])], axis=1)
        df_subset = df_subset.iloc[:, [randint(0, df_subset.shape[1]) for _ in range(10)]]
        print('-----Getting Naive Bayes Predictions-----')
        df['Naive_Bayes_predictions'] = gnb.predict(df_subset)
        df['Naive_Bayes_confidence'] = gnb.predict_proba(df_subset)

        print('-----Getting KNN Predictions-----')
        df['KNN_predictions'] = neigh.predict(df_subset)
        df['KNN_predictions_confidence'] = neigh.predict_proba(df_subset)

        print('-----Getting Random Forest Predictions-----')
        df['Random_Forest_predictions'] = forest.predict(df_subset)
        df['Random_Forest_predictions_confidence'] = forest.predict_proba(df_subset)

        df.loc[:, ['doi', 'Naive_Bayes_predictions', 'KNN_predictions', 'Random_Forest_predictions']].to_excel('data/covid19_kb/{}_output.xlsx'.format(i))
