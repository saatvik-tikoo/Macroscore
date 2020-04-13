import pandas as pd
import pickle
import json

my_dict = dict()
with open('data/doi_to_id_mappings.json') as infile:
    data = json.load(infile)
    for obj in data:
        my_dict[obj['id']] = obj['doi']

with open('data/doi_id_dict.pickle', 'wb') as handle:
    pickle.dump(my_dict, handle)

embedding_filenames = ['data/node2vec_references_network_2hops.emb']
final_filenames = ['data/finalData.xlsx']
final_data = []
with open(embedding_filenames[0]) as f_read:
    lines = f_read.readlines()
    print('-----Generating Dictionary-----')
    for idx_l, line in enumerate(lines):
        if idx_l > 0:
            vals = line.split(' ')
            if vals[0] in my_dict:
                cur_obj = {
                    'doi': my_dict[vals[0]]
                }
                for _, col in enumerate(vals[1:]):
                    cur_obj['new_feature_' + str(_ + 1)] = col
                final_data.append(cur_obj)

df = pd.DataFrame(final_data)
print('-----Saving File-----')
df.dropna(subset=['new_feature_1'], inplace=True)
print(df.shape)
df.to_excel(final_filenames[0])
