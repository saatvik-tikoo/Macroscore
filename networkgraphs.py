import itertools
import os
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

class NetworkGraph:
    def __init__(self):
        self.path_head = '../DataExtraction/WOS/RPPdataConverted'
        self.df = None

    def get_data(self, features=['DOI', 'Study.Title.O']):
        if features == 'all':
            self.df = pd.read_excel('data/new_data.xlsx', encoding='ansi',)
        else:
            self.df = pd.read_excel('data/new_data.xlsx', encoding='ansi', usecols=features)

    def citation_graph(self):
        g = nx.Graph()
        nodes = dict()
        self.df.dropna(subset=['DOI'])
        for idx, row in self.df.iterrows():
            print('Row id: ', idx)
            name = row['DOI'].replace('/', '_')
            file_name = self.path_head + '/{}/citations_{}.txt'.format(name, name)
            if os.path.exists(file_name):
                if name not in nodes and row['Study.Title.O'] not in nodes.values():
                    g.add_node(name)
                    nodes[name] = row['Study.Title.O']
                elif row['Study.Title.O'] in nodes.values():
                    del nodes[row('Study.Title.O')]
                    nodes[name] = row['Study.Title.O']
                    g.remove_node(row('Study.Title.O'))
                    g.add_node(name)
                df_doi = pd.read_csv(file_name, sep='\t', lineterminator='\r', encoding="utf-16le",
                                     index_col=False, quotechar=None, quoting=3, usecols=['DI', 'TI'])
                df_doi = df_doi.dropna()
                for i, citation_row in df_doi.iterrows():
                    if citation_row['DI']:
                        if citation_row['DI'] not in nodes and citation_row['TI'] not in nodes:
                            g.add_node(citation_row['DI'])
                            nodes[citation_row['DI']] = citation_row['TI']
                        elif citation_row['TI'] in nodes:
                            del nodes[citation_row['TI']]
                            nodes[citation_row['DI']] = citation_row['TI']
                            g.remove_node(citation_row['TI'])
                            g.add_node(citation_row['DI'])
                            g.remove_edge(citation_row['TI'], name)
                        g.add_edge(name, citation_row['DI'])
                    elif citation_row['TI']:
                        if citation_row['TI'] not in nodes:
                            g.add_node(citation_row['TI'])
                            nodes[citation_row['TI']] = citation_row['TI']
                        g.add_edge(name, citation_row['TI'])
        print('-------------Saving Graph--------------')
        print('Number of nodes in the graph: ', len(g.nodes()))
        print('Number of edges in the graph: ', len(g.edges()))
        nx.write_gpickle(g, 'data/citations_network.gpickle')
        nx.write_gexf(g, "data/citations_network.gexf")
        print('-------------Graph Saved--------------')

    def coauthorship_graph(self):
        g = nx.Graph()
        nodes = set()
        self.df.dropna(subset=['DOI'])
        for idx, row in self.df.iterrows():
            print('Row id: ', idx)
            name = row['DOI'].replace('/', '_')
            folder_name = self.path_head + '/{}/'.format(name)
            if os.path.exists(folder_name):
                for file_name in os.listdir(folder_name):
                    if file_name.startswith("authors_"):
                        df_doi = pd.read_csv(folder_name + '/' + file_name, sep='\t', lineterminator='\r', encoding="utf-16le",
                                             index_col=False, quotechar=None, quoting=3, usecols=['AU'])
                        df_doi.dropna()
                        co_authors = dict()
                        for _, authors_row in df_doi.iterrows():
                            cur_auths = authors_row['AU'].split(';')
                            cur_auths = [x.strip().lower() for x in cur_auths]
                            for au in cur_auths:
                                if au in co_authors:
                                    co_authors[au] += 1
                                else:
                                    co_authors[au] = 1
                        main_author = max(co_authors, key=co_authors.get)
                        if main_author not in nodes:
                            g.add_node(main_author)
                            nodes.add(main_author)
                        for _, authors_row in df_doi.iterrows():
                            cur_auths = authors_row['AU'].split(';')
                            cur_auths = [x.strip().lower() for x in cur_auths]
                            for au in cur_auths:
                                if au not in nodes:
                                    nodes.add(au)
                                if main_author != au:
                                    g.add_edge(main_author, au)
                            for au_comb in itertools.combinations(cur_auths, 2):
                                if au not in nodes:
                                    nodes.add(au)
                                if main_author != au:
                                    g.add_edge(au_comb[0], au_comb[1])
        print('-------------Saving Graph--------------')
        print('Number of nodes in the graph: ', len(g.nodes()))
        print('Number of edges in the graph: ', len(g.edges()))
        nx.write_gpickle(g, 'data/coauthorship_network.gpickle')
        nx.write_gexf(g, "data/coauthorship_network.gexf")
        print('-------------Graph Saved--------------')

    def display_graph(self, file):
        print('-------------Generating Graph----------')
        g = nx.read_gpickle(file)
        print('Number of nodes in the graph: ', len(g.nodes()))
        print('Number of edges in the graph: ', len(g.edges()))
        nx.draw_networkx(G=g, pos=nx.spring_layout(g), node_color='r', alpha=0.8,
                         node_size=[g.degree(n) * 3 for n in g.nodes()], with_labels=False)
        plt.show()

    def addNetworkFeatures(self):
        with open('data/node2vec_citations.emb', 'r') as f_read:
            lines = f_read.readlines()
            for row in range(len(lines)):
                if row > 0:
                    vals = lines[row].split(' ')
                    doi = vals[0]
                    # Check for Titles instead of DOIS too
                    if doi[0] == '1':
                        doi = doi.replace('_', '/')
                        print('--------------------DOI: ', doi, ' and row: ', row + 1, '--------------------')
                        for idx in range(len(vals) - 1):
                            self.df.loc[self.df['DOI'] == doi, 'new_feature_' + str(idx + 1)] = vals[idx + 1]
        self.df.to_excel('data/final_network_data.xlsx')


if __name__ == '__main__':
    cnn = NetworkGraph()
    # Uncomment below to generate the graph: Step-2
    # cnn.get_data()
    # cnn.citation_graph()
    # cnn.coauthorship_graph()

    # Uncomment below to add the generated node2vec features to our data: Step-4
    cnn.get_data('all')
    cnn.addNetworkFeatures()

    # Uncomment below to display the generated graph
    # cnn.display_graph('data/citations_network.gpickle')
