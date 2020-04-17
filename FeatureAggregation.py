import os
from collections import defaultdict
import pandas as pd

Initial_path = 'C:\\Users\\Saatvik\\Desktop\\MacroScope\\ClassifierDev\\data\\rpp_data\\'

filename = [f for f in os.listdir(Initial_path)]

df2 = pd.read_excel('data/RPPdata.xlsx')
df2.dropna(subset=['DOI'], inplace=True)
mappings = dict()
labels = dict()
authors = dict()
for i, rows in df2.iterrows():
    idx = str(int(str(rows['Local.ID']).strip().split('.')[-1]) - 1)
    mappings[idx] = rows['DOI']
    labels[idx] = rows['Meta.analysis.significant']
    authors[rows['DOI']] = rows['Authors.O']

data = defaultdict(list)
for _, file in enumerate(filename):
    with open(Initial_path + file) as f_read:
        lines = f_read.readlines()
        for line in lines:
            vals = line.strip().split('\t')
            if vals[0] in mappings:
                data[mappings[vals[0]]].append(vals[-1])
                if _ == len(filename) - 1:
                    data[mappings[vals[0]]].append(labels[vals[0]])

dataset = []
for k, v in data.items():
    dataset.append({
        'doi': k,
        'Authors.O': authors[k],
        'feature_1': v[0],
        'feature_2': v[1],
        'feature_3': v[2],
        'feature_4': v[3],
        'feature_5': v[4],
        'feature_6': v[5],
        'feature_7': v[6],
        'feature_8': v[7],
        'label': v[8]
    })

df = pd.DataFrame(dataset)
df.dropna(inplace=True)
print(df.columns.values)
df.to_excel('data/XeuanFeatures.xlsx')