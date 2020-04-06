import rdflib as rdflib
import networkx as nx

def create_2_hop_graph(g, root, depth, nx_graph):
    to_explore = {root}
    for hop in range(depth):
        print('Hop: ', hop)
        new_explore = set()
        for node in to_explore:
            for s, p, o in g.triples((node, None, None)):
                if 'cites' in str(p).lower():
                    s_name = str(s).split('/')[-1]
                    o_name = str(o).split('/')[-1]
                    nx_graph.add_edge(s_name, o_name)
                    new_explore.add(o)
        to_explore = new_explore
    return nx_graph

def create_graph(g, nx_graph):
    for s, p, o in g:
        if 'cites' in str(p).lower():
            s_name = str(s).split('/')[-1]
            o_name = str(o).split('/')[-1]
            nx_graph.add_edge(s_name, o_name)
    return nx_graph


if __name__ == '__main__':
    g_kb = rdflib.Graph()
    g_kb.parse('data/covid19_kb/kg.nt', format='nt')
    generated_graph = nx.DiGraph()

    generated_graph = create_graph(g_kb, generated_graph)
    print('---------Getting Degree---------')
    degree_dict = dict(generated_graph.degree(generated_graph.nodes()))
    nx.set_node_attributes(generated_graph, degree_dict, 'degree')

    print('-------------Saving Graph--------------')
    print('Number of nodes in the graph: ', len(generated_graph.nodes()))
    print('Number of edges in the graph: ', len(generated_graph.edges()))
    nx.write_gpickle(generated_graph, 'data/covid19_kb/references_network_whole.gpickle')
    nx.write_gexf(generated_graph, "data/covid19_kb/references_network_whole.gexf")
    print('-------------Graph Saved--------------')

    for sub, _, _ in g_kb:
        print('-----For ', str(sub).split('/')[-1][:25], '-----')
        generated_graph = create_2_hop_graph(g_kb, sub, 2, generated_graph)

    print('-------------Saving Graph--------------')
    print('Number of nodes in the graph: ', len(generated_graph.nodes()))
    print('Number of edges in the graph: ', len(generated_graph.edges()))
    nx.write_gpickle(generated_graph, 'data/covid19_kb/references_network_2hops.gpickle')
    nx.write_gexf(generated_graph, "data/covid19_kb/references_network_2hops.gexf")
    print('-------------Graph Saved--------------')
