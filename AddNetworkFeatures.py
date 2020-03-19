import json
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


class NetworkFetaures:
    def __init__(self, file, graph_type):
        self.df = None
        self.file = file
        self.graph_type = graph_type

    def get_data(self, features=['DOI']):
        self.df = pd.read_excel('data/new_data.xlsx', encoding='ansi', )

    def addNetworkFeatures(self, graph_type='wos'):
        g = nx.read_gpickle('data/citations_network_2hops_wos.gpickle')
        with open(self.file, 'r') as f_read:
            lines = f_read.readlines()
            if graph_type == 'mag':
                mappings = json.loads(json.dumps(pickle.load(open("data/IDS_doi_mapping", "rb"))))
            lines_dict = dict()
            for idx_l, line in enumerate(lines):
                if idx_l > 0:
                    vals = line.split(' ')
                    if graph_type == 'wos':
                        lines_dict[vals[0]] = vals[1:]
                    elif graph_type == 'mag':
                        lines_dict[mappings[vals[0]]] = vals[1:]

            for _, row in self.df.iterrows():
                doi = row['DOI'].replace('/', '_')
                idx = 0
                if doi in lines_dict:
                    fields = lines_dict[doi]
                    for idx, col in enumerate(fields):
                        self.df.at[_, 'new_feature_' + str(idx + 1)] = col
                print('--------------------DOI: {} and row: {} and total-cols are: {}--------------------'.format(doi, _ + 1, idx + 1))
        print(self.df.shape)
        self.df.to_excel('data/final_network_data.xlsx')
        print('Done')

    def display_graph(self):
        print('-------------Generating Graph----------')
        g = nx.read_gpickle(self.file)
        print('Number of nodes in the graph: ', len(g.nodes()))
        print('Number of edges in the graph: ', len(g.edges()))
        nx.draw_networkx(G=g, pos=nx.spring_layout(g), node_color='r', alpha=0.8,
                         node_size=[g.degree(n) * 3 for n in g.nodes()], with_labels=False)
        plt.show()


if __name__ == '__main__':
    # Set a proper file name and graph_type = 'wos' or 'mag'
    cnn = NetworkFetaures(file='data/node2vec_citations_network_2hops_wos.emb', graph_type='wos')
    cnn.get_data()
    cnn.addNetworkFeatures()

    # Uncomment below to display the generated graph
    # cnn.display_graph('data/citations_network_2hops_wos.gpickle')
