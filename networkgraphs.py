import itertools
import json
import os
import pickle

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

    def __citation_graph_wos__(self, g, DOIS, hops):
        if hops == 0:
            return
        new_dois = set()
        for doi in DOIS:
            name = doi.replace('/', '_')
            file_name = self.path_head + '/{}/citations_{}.txt'.format(name, name)
            if os.path.exists(file_name):
                df_doi = pd.read_csv(file_name, sep='\t', lineterminator='\r', encoding="utf-16le",
                                     index_col=False, quotechar=None, quoting=3, usecols=['DI'])
                df_doi = df_doi.dropna()
                for i, citation_row in df_doi.iterrows():
                    if not g.has_edge(name, citation_row['DI']):
                        new_dois.add(citation_row['DI'])
                        g.add_edge(name.replace('/', '_'), citation_row['DI'])
        self.__citation_graph_wos__(g, new_dois, hops - 1)

    def citation_graph_wos(self, hops=2):
        # Based on WOS data
        g = nx.DiGraph()
        self.df.dropna(subset=['DOI'])
        self.__citation_graph_wos__(g, list(self.df['DOI']), hops)

        betweenness_dict = nx.betweenness_centrality(g)
        nx.set_node_attributes(g, betweenness_dict, 'betweenness')

        degree_dict = dict(g.degree(g.nodes()))
        nx.set_node_attributes(g, degree_dict, 'degree')

        print('-------------Saving Graph--------------')
        print('Number of nodes in the graph: ', len(g.nodes()))
        print('Number of edges in the graph: ', len(g.edges()))
        nx.write_gpickle(g, 'data/citations_network_{}hops_wos.gpickle'.format(hops))
        nx.write_gexf(g, "data/citations_network_{}hops_wos.gexf".format(hops))
        print('-------------Graph Saved--------------')

    def __citation_graph_mag__(self, g, IDS, df_citations, graph_type, hops):
        print('---------Running for {} hop---------'.format(hops))
        if hops == 0:
            return
        new_IDS = []
        for cur_id in IDS:
            if graph_type == 'references':
                print('References are getting fetched for: ', cur_id)
                new_nodes = list(df_citations.loc[df_citations['from'] == cur_id]['to'])
            else:
                print('Citations are getting fetched for: ', cur_id)
                new_nodes = list(df_citations.loc[df_citations['to'] == cur_id]['from'])
            for edge_node in new_nodes:
                if not g.has_edge(cur_id, edge_node):
                    g.add_edge(cur_id, edge_node)
            new_IDS.extend(new_nodes)
        print('ids for next hop ', len(new_IDS))
        self.__citation_graph_mag__(g, set(new_IDS), df_citations, graph_type, hops - 1)

    def citation_graph_mag(self, graph_type='references', hops=2):
        # Based on MAG data
        g = nx.DiGraph()
        self.df.dropna(subset=['DOI'])
        print('---------Getting all the edges---------')
        df_citations = pd.read_csv("data/MAG_data/psychology_citations.csv")
        # map all the dois to an id and send that
        IDS = dict()
        mappings = json.loads(json.dumps(pickle.load(open("data/MAG_data/rpp.pkl", "rb"))))
        for doi in list(self.df['DOI']):
            if doi in mappings:
                IDS[mappings[doi][0]] = doi
        self.__citation_graph_mag__(g, IDS.keys(), df_citations, graph_type, 2)
        print('---------Getting betweenness---------')
        betweenness_dict = nx.betweenness_centrality(g)
        nx.set_node_attributes(g, betweenness_dict, 'betweenness')
        print('---------Getting Degree---------')
        degree_dict = dict(g.degree(g.nodes()))
        nx.set_node_attributes(g, degree_dict, 'degree')

        print('-------------Saving Graph--------------')
        print('Number of nodes in the graph: ', len(g.nodes()))
        print('Number of edges in the graph: ', len(g.edges()))
        nx.write_gpickle(g, 'data/{}_network_{}hops_mag.gpickle'.format(graph_type, hops))
        nx.write_gexf(g, "data/{}_network_{}hops_mag.gexf".format(graph_type, hops))
        pickle.dump(IDS, open('data/IDS_doi_mapping.pkl', 'wb'))
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

    def addNetworkFeatures(self, graph_type='wos'):
        with open('data/node2vec_citations.emb', 'r') as f_read:
            lines = f_read.readlines()
            if graph_type == 'mag':
                mappings = json.loads(json.dumps(pickle.load(open("data/IDS_doi_mapping", "rb"))))
            lines_dict = dict()
            for idx_l, line in enumerate(lines):
                if idx_l > 0:
                    vals = line.split(' ')
                    if graph_type == 'wos':
                        lines_dict[vals[0]] = vals[1:]
                    else:
                        lines_dict[mappings[vals[0]]] = vals[1:]

            for _, row in self.df.iterrows():
                doi = row['DOI'].replace('/', '_')
                print('--------------------DOI: ', doi, ' and row: ', _ + 1, '--------------------')
                if doi in lines_dict:
                    fields = lines_dict[doi]
                    for idx, col in enumerate(fields):
                        row['new_feature_' + str(idx + 1)] = col
        self.df.to_excel('data/final_network_data.xlsx')


if __name__ == '__main__':
    cnn = NetworkGraph()
    # Uncomment below to generate the graph: Step-2
    cnn.get_data()
    cnn.citation_graph_mag('references', hops=2)
    # cnn.coauthorship_graph()

    # Uncomment below to add the generated node2vec features to our data: Step-4
    # cnn.get_data('all')
    # cnn.addNetworkFeatures('wos')

    # Uncomment below to display the generated graph
    # cnn.display_graph('data/citations_network.gpickle')
