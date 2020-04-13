import pandas as pd
import numpy as np
import networkx as nx
import json
from collections import defaultdict
import pickle

df_citations = pd.read_csv('data/covid_citations.csv', usecols=['from', 'to'])
df_papers = pd.read_csv('data/covid_papers.csv', usecols=['DOI', 'RId', 'Id'])
df_metadata = pd.read_csv('data/metadata.csv', usecols=['doi'])


def get_mappings():
    mappings = []
    cnt = 0
    check_list = ['10.1016/j.eng.2020.03.007', '10.3760/cma.j.cn112338-20200221-00144', '10.1128/JVI.05050-11',
                  '10.1128/JVI.01570-14', '10.1016/S2214-109X(20)30065-6', '10.1101/2020.01.30.927871']
    for i, row in df_metadata.iterrows():
        paper_id = df_papers.loc[str(row['doi']).strip() == df_papers['DOI'], 'Id'].values
        if row['doi'] in check_list:
            print(row['doi'], ' : ', paper_id)
        if len(paper_id) > 0:
            cnt += 1
            mappings.append({
                'doi': row['doi'],
                'id': paper_id[0]
            })
    print(cnt)
    with open('data/doi_to_id_mappings.json', 'w') as outfile:
        json.dump(mappings, outfile)


get_mappings()

from_to_dict = defaultdict(list)
for i, row in df_citations.iterrows():
    from_to_dict[row['from']].append(row['to'])

with open('data/from_to_dict.pickle', 'wb') as handle:
    pickle.dump(from_to_dict, handle)

def generate_graph(hops=2):
    g = nx.DiGraph()
    new_nodes = []
    with open('data/doi_to_id_mappings.json') as infile:
        data = json.load(infile)
        for _ in range(hops):
            for obj in data:
                if obj['id'] in from_to_dict:
                    for link in from_to_dict[obj['id']]:
                        g.add_edge(obj['id'], link)
                        new_nodes.append({'id': link})
            data = list(new_nodes)

    print('-------------Saving Graph--------------')
    print('Number of nodes in the graph: ', len(g.nodes()))
    print('Number of edges in the graph: ', len(g.edges()))
    nx.write_gpickle(g, 'data/references_network_2hops.gpickle')
    print('-------------Graph Saved--------------')

generate_graph(hops=2)
