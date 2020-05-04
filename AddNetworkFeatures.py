import json
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


class NetworkFetaures:
    def __init__(self, file, output, graph_type):
        self.df = None
        self.file = file
        self.output = output
        self.graph_type = graph_type

    def get_data(self):
        self.df = pd.read_excel('data/new_data.xlsx', encoding='ansi', usecols=['DOI', 'P.value.R', 'Direction.R',
                                                                                'O.within.CI.R', 'Authors.O',
                                                                                'Meta.analysis.significant', 'Fold_Id'])

    def addNetworkFeatures(self):
        with open(self.file, 'r') as f_read:
            lines = f_read.readlines()
            mappings = None
            if self.graph_type == 'mag_c':
                mappings = json.loads(json.dumps(pickle.load(open("data/IDS_doi_mapping_citations.pkl", "rb"))))
            elif self.graph_type == 'mag_r':
                mappings = json.loads(json.dumps(pickle.load(open("data/IDS_doi_mapping_references.pkl", "rb"))))
            lines_dict = dict()
            for idx_l, line in enumerate(lines):
                if idx_l > 0:
                    vals = line.split(' ')
                    if 'wos' in self.graph_type:
                        lines_dict[vals[0]] = vals[1:]
                    elif 'mag' in self.graph_type and vals[0] in mappings:
                        lines_dict[mappings[vals[0]]] = vals[1:]

            for _, row in self.df.iterrows():
                doi = row['DOI']
                idx = 0
                if doi in lines_dict:
                    fields = lines_dict[doi]
                    for idx, col in enumerate(fields):
                        self.df.at[_, 'new_feature_' + str(idx + 1) + self.graph_type] = col
                print('--------------------DOI: {} and row: {} and total-cols are: {}--------------------'.format(doi, _ + 1, idx + 1))
        print(self.df.shape)
        self.df.to_excel(self.output)
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
    # Set a proper file name and graph_type = 'wos' or 'mag_c' or 'mag_r'
    input_file = ['data/node2vec_references_network_2hops_wos.emb', 'data/node2vec_references_network_2hops_mag.emb',
                  'data/node2vec_citations_network_2hops_mag.emb']
    output_file = ['data/final_references_wos_data.xlsx', 'data/final_references_mag_data.xlsx',
                   'data/final_citations_mag_data.xlsx']
    graph = ['wos', 'mag_r', 'mag_c']
    for i in range(3):
        cnn = NetworkFetaures(file=input_file[i], output=output_file[i], graph_type=graph[i])
        cnn.get_data()
        cnn.addNetworkFeatures()

    # Uncomment below to display the generated graph
    # cnn.display_graph('data/citations_network_2hops_wos.gpickle')
