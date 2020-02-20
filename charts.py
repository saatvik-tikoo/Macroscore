import numpy as np
import seaborn as sns

import macroscore
from matplotlib import pyplot as plt, rcParams
from sklearn.metrics import auc, roc_curve, accuracy_score


def clean_data_local():
    # Get required fields of the data-set (fields not of the Replication Study)
    req_columns = ['P-value (R)', 'Direction (R)', 'Meta-analysis significant', 'O within CI R']
    remove_columns = ['Study Title (O)', 'Descriptors (O)', 'Description of effect (O)', 'Feasibility (O)',
                      'Calculated P-value (O)', 'Test statistic (O)', 'Type of analysis (O)']
    data_types = ['Integer', 'Range', 'Categorical', 'Decimal', 'Dichotomous', 'Open text', 'Mixed']
    with open("Psychology/rpp_data_codebook.csv") as input_file:
        reader = macroscore.csv.reader(input_file)
        # skip the header
        next(reader)
        for line in reader:
            if '(R)' not in line[0] and line[4] in data_types and line[0] not in remove_columns:
                req_columns.append(line[0])
    data_set = macroscore.pandas.read_csv('Psychology/rpp_data_updates.csv', encoding='ansi', usecols=req_columns)

    # Add output field and dropping the Replication fields
    data_set['result_label'] = data_set.apply(lambda row: macroscore.label_addition(row), axis=1)
    data_set = data_set.drop(['P-value (R)', 'Direction (R)'], axis=1)

    # Clean fields
    data_set = data_set.replace(to_replace='X', value=macroscore.np.nan)
    # data_set = data_set.dropna(subset=['Effect size (O)', 'Reported P-value (O)', 'Meta-analysis significant', 'O within CI R'])
    data_set = data_set.dropna(subset=['Institution of senior author (O)'])
    data_set = data_set.replace(to_replace=macroscore.np.nan, value="0")
    data_set = macroscore.clean_pvalue(data_set, 'Reported P-value (O)')
    data_set = macroscore.clean_range_values(data_set, 'Pages (O)')
    data_set = macroscore.clean_float_result(data_set, 'Surprising result (O)')
    data_set = macroscore.clean_float_result(data_set, 'Exciting result (O)')
    data_set = macroscore.clean_effect_size(data_set, 'Effect size (O)')

    # Removing the last row as it is a null row
    data_set = data_set[: -1]

    # Changing the data-types of some fields so as not to include them in encoding
    data_set = data_set.astype({'Reported P-value (O)': 'float64', 'Pages (O)': 'float64', 'Surprising result (O)': 'float64',
                                'Exciting result (O)': 'float64', 'N (O)': 'float64', 'Effect size (O)': 'float64', '# Tails (O)': 'float64',
                                'Meta-analysis significant': 'bool', 'O within CI R': 'bool'})

    print(data_set.shape)
    return data_set


def plot_charts(data):
    fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.show()

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
    df = clean_data_local()
    plot_charts(df)
