import os
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import pickle
import json
import numpy as np

SYNTHETIC_EDGES_SIZE = 0
class NetworkGraph:
    def __init__(self):
        self.path_head = '../DataExtraction/WOS/RPPdataConverted'
        self.df = None

    def get_data(self, features=['DOI', 'Study.Title.O']):
        self.df = pd.read_excel('data/new_data.xlsx', encoding='ansi', usecols=features)

    def __graph_wos__(self, g, DOIS, hops, reproducible):
        if hops == 0:
            return
        print('---------Running for {} hop---------'.format(hops))
        new_dois = set()
        for idx_doi, doi in enumerate(DOIS):
            name = doi.replace('/', '_')
            file_name = self.path_head + '/{}/citations_{}.txt'.format(name, name)
            if os.path.exists(file_name):
                print('References are getting fetched for: ', idx_doi, name)
                df_doi = pd.read_csv(file_name, sep='\t', lineterminator='\r', encoding="utf-16le",
                                     index_col=False, quotechar=None, quoting=3, usecols=['DI'])
                df_doi = df_doi.dropna()
                for i, citation_row in df_doi.iterrows():
                    if not g.has_edge(name, citation_row['DI']):
                        new_dois.add(citation_row['DI'])
                        g.add_edge(doi, citation_row['DI'])

            # Adding Synthetic edges
            if hops == 2 and doi not in reproducible:
                randoms = np.random.randint(len(reproducible), size=SYNTHETIC_EDGES_SIZE)
                for ppr_idx in randoms:
                    g.add_edge(doi, reproducible[ppr_idx])

        print('Number of DOIs for next hop ', len(new_dois))
        self.__graph_wos__(g, new_dois, hops - 1, reproducible)

    def graph_wos(self, reproducible, hops=2):
        # Based on WOS data
        g = nx.DiGraph()
        self.df.dropna(subset=['DOI'])
        self.__graph_wos__(g, set(self.df['DOI']), hops, reproducible)
        # print('---------Getting betweenness---------')
        # betweenness_dict = nx.betweenness_centrality(g)
        # nx.set_node_attributes(g, betweenness_dict, 'betweenness')
        print('---------Getting Degree---------')
        degree_dict = dict(g.degree(g.nodes()))
        nx.set_node_attributes(g, degree_dict, 'degree')
        #
        print('-------------Saving Graph--------------')
        print('Number of nodes in the graph: ', len(g.nodes()))
        print('Number of edges in the graph: ', len(g.edges()))
        nx.write_gpickle(g, 'data/references_network_{}hops_wos.gpickle'.format(hops))
        nx.write_gexf(g, "data/references_network_{}hops_wos.gexf".format(hops))
        print('-------------Graph Saved--------------')

    def __graph_mag__(self, g, IDS, df_citations, graph_type, hops, reproducible):
        if hops == 0:
            return
        print('---------Running for {} hop---------'.format(hops))
        new_IDS = []
        for idx, cur_id in enumerate(IDS):
            if graph_type == 'references':
                print('References are getting fetched for: ', idx, ' ', cur_id)
                new_nodes = list(df_citations.loc[df_citations['from'] == cur_id]['to'])
            elif graph_type == 'citations':
                print('Citations are getting fetched for: ', idx, ' ', cur_id)
                new_nodes = list(df_citations.loc[df_citations['to'] == cur_id]['from'])
            for edge_node in new_nodes:
                if not g.has_edge(cur_id, edge_node):
                    g.add_edge(cur_id, edge_node)
            new_IDS.extend(new_nodes)

            # Adding Synthetic edges
            if hops == 2 and cur_id not in reproducible:
                randoms = np.random.randint(len(reproducible), size=SYNTHETIC_EDGES_SIZE)
                for ppr_idx in randoms:
                    g.add_edge(cur_id, reproducible[ppr_idx])
        print('Number of DOIs for next hop ', len(new_IDS))
        self.__graph_mag__(g, set(new_IDS), df_citations, graph_type, hops - 1, reproducible)

    def graph_mag(self, reproducible, graph_type='references', hops=2):
        # Based on MAG data
        g = nx.DiGraph()
        self.df.dropna(subset=['DOI'])
        print('---------Getting all the edges---------')
        df_citations = pd.read_csv("data/MAG_data/psychology_citations.csv")
        # map all the dois to an id and send that
        IDS = dict()
        mappings = json.loads(json.dumps(pickle.load(open("data/MAG_data/rpp.pkl", "rb"))))
        reproducible_ids = []
        for doi in list(self.df['DOI']):
            if doi in mappings:
                IDS[mappings[doi][0]] = doi
                # Get ids for reproducible papers to generate synthetic edges
                if doi in reproducible:
                    reproducible_ids.append(mappings[doi][0])

        self.__graph_mag__(g, IDS.keys(), df_citations, graph_type, 2, reproducible_ids)
        # print('---------Getting betweenness---------')
        # betweenness_dict = nx.betweenness_centrality(g)
        # nx.set_node_attributes(g, betweenness_dict, 'betweenness')
        print('---------Getting Degree---------')
        degree_dict = dict(g.degree(g.nodes()))
        nx.set_node_attributes(g, degree_dict, 'degree')

        print('-------------Saving Graph--------------')
        print('Number of nodes in the graph: ', len(g.nodes()))
        print('Number of edges in the graph: ', len(g.edges()))
        nx.write_gpickle(g, 'data/{}_network_{}hops_mag.gpickle'.format(graph_type, hops))
        nx.write_gexf(g, "data/{}_network_{}hops_mag.gexf".format(graph_type, hops))
        pickle.dump(IDS, open('data/IDS_doi_mapping_{}.pkl'.format(graph_type), 'wb'))
        print('-------------Graph Saved--------------')

    def display_graph(self, file):
        print('-------------Generating Graph----------')
        g = nx.read_gpickle(file)
        print('Number of nodes in the graph: ', len(g.nodes()))
        print('Number of edges in the graph: ', len(g.edges()))
        nx.draw_networkx(G=g, pos=nx.spring_layout(g), node_color='r', alpha=0.8,
                         node_size=[g.degree(n) * 3 for n in g.nodes()], with_labels=False)
        plt.show()

def separate_papers():
    df = pd.read_excel('data/new_data.xlsx')
    reproducible = [row['DOI'] for idx, row in df.iterrows() if not pd.isna(row['DOI']) and row['pvalue.label'] == 1]
    return reproducible


if __name__ == '__main__':
    cnn = NetworkGraph()
    rep = separate_papers()
    cnn.get_data()
    # WOS data has only option of getting references. So only option we can change is number of hops
    # cnn.graph_wos(reproducible=rep, hops=2)

    # Mag data has two options for generating the graph_type='references' and 'citations', Also we can set the number of hops
    cnn.graph_mag(reproducible=rep, graph_type='references', hops=2)
