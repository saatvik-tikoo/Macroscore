import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, rcParams
import pandas as pd
from sklearn.metrics import auc, roc_curve, accuracy_score
from sklearn import manifold

def plot_charts(data):
    cols_drop = set(['P.value.R', 'Direction.R', 'O.within.CI.R', 'Meta.analysis.significant', 'pvalue.label', 'DOI', '1st.author.O', 'Senior.author.O', 'Authors.O', 'Study.Title.O', 'Unnamed: 0', 'new_feature_301', 'Unnamed: 0.1'])
    cols_total = set(data.columns)
    X = data.drop(cols_drop.intersection(cols_total), axis=1)
    X = X.dropna()
    y = data['O.within.CI.R']
    X_norm = (X - X.min()) / (X.max() - X.min())
    pca = manifold.TSNE(n_components=2)  # 2-dimensional PCA
    transformed = pd.DataFrame(pca.fit_transform(X_norm))
    plt.scatter(transformed[y == 0][0], transformed[y == 0][1], label='Non-Replicable', c='red')
    plt.scatter(transformed[y == 1][0], transformed[y == 1][1], label='Replicable', c='blue')

    plt.legend()

    # sns.heatmap(data.corr(), annot=True, fmt=".2f")
    # grid = sns.catplot(x="Institution of senior author (O)", kind="count", palette="pastel", edgecolor=".6", data=data)
    # plt.xticks(fontsize=8, rotation=90)
    # plt.show()
    # df_pivot = data.groupby(['result_label', 'Institution of 1st author (O)']).size().reset_index()\
    #     .pivot(columns='result_label', index='Institution of 1st author (O)', values=0)
    # df_pivot = df_pivot.reindex().sort_index(axis=1)
    # df_pivot.plot.bar(stacked=True)
    # dis = data['Institution of senior author (O)']
    # sns.distplot(dis)
    plt.show()


if __name__ == '__main__':
    fileName = 'data/final_network_data.xlsx'
    df = pd.read_excel(fileName, encoding='ansi')
    plot_charts(df)
