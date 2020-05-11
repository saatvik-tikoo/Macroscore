import numpy as np
import networkx as nx
import random
from gensim.models import Word2Vec


class Node2Vec:
    def __init__(self, g, p=1, q=1, num_walks=10, walk_length=10, dimensions=300,
                 window_size=10, workers=8, iterations=1, output=None):
        self.G = g
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = workers
        self.iterations = iterations
        self.output = output
        self.alias_nodes = dict()
        self.alias_edges = dict()

    def __alias_setup__(self, probs):
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        return J, q

    def __get_alias_edge__(self, src, dst):
        unnormalized_probs = []
        for dst_nbr in sorted(self.G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(self.G[dst][dst_nbr]['weight'] / self.p)
            elif self.G.has_edge(dst_nbr, src):
                unnormalized_probs.append(self.G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(self.G[dst][dst_nbr]['weight'] / self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return self.__alias_setup__(normalized_probs)

    def __alias_draw__(self, J, q):
        K = len(J)

        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def __node2vec_walk__(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[self.__alias_draw__(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    nxt = cur_nbrs[self.__alias_draw__(self.alias_edges[(prev, cur)][0], self.alias_edges[(prev, cur)][1])]
                    walk.append(nxt)
            else:
                break
        return walk

    def preprocess_transition_probs(self):
        for node in self.G.nodes():
            unnormalized_probs = [self.G[node][nbr]['weight'] for nbr in sorted(self.G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[node] = self.__alias_setup__(normalized_probs)

        for edge in self.G.edges():
            self.alias_edges[edge] = self.__get_alias_edge__(edge[0], edge[1])
            # self.alias_edges[(edge[1], edge[0])] = self.__get_alias_edge__(edge[1], edge[0])
        return

    def simulate_walks(self):
        walks = []
        nodes = list(self.G.nodes())
        print('Walk iteration:')
        for walk_iter in range(self.num_walks):
            print(walk_iter + 1, ' out of ', self.num_walks)
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.__node2vec_walk__(start_node=node))
        return walks

    def learn_embeddings(self, walks):
        model = Word2Vec(walks, size=self.dimensions, window=self.window_size,
                         min_count=0, sg=1, workers=self.workers, iter=self.iterations)
        model.wv.save_word2vec_format(self.output)
        return

def read_graph(file_name):
    g = nx.read_gpickle(file_name)
    for edge in g.edges():
        g[edge[0]][edge[1]]['weight'] = 1
    return g


if __name__ == "__main__":
    # Add the input and output files here in the same sequence.
    input_file = ['data/references_network_2hops_wos_1.gpickle',
                  'data/references_network_2hops_wos_10.gpickle',
                  'data/references_network_2hops_mag_synthetic_1.gpickle',
                  'data/references_network_2hops_mag_synthetic_10.gpickle',
                  'data/citations_network_2hops_mag_synthetic_1.gpickle',
                  'data/citations_network_2hops_mag_synthetic_10.gpickle']
    output_file = ['data/node2vec_references_network_2hops_wos_synthetic_1.emb',
                   'data/node2vec_references_network_2hops_wos_synthetic_10.emb',
                   'data/node2vec_references_network_2hops_mag_synthetic_1.emb',
                   'data/node2vec_references_network_2hops_mag_synthetic_10.emb',
                   'data/node2vec_citations_network_2hops_mag_synthetic_1.emb',
                   'data/node2vec_citations_network_2hops_mag_synthetic_10.emb'
                   ]
    for i in range(len(input_file)):
        print('----------Getting Features for file ', i + 1, '----------')
        nx_G = read_graph(input_file[i])
        n2v = Node2Vec(nx_G, p=2, q=2, output=output_file[i])
        print('----Preprocessing----')
        n2v.preprocess_transition_probs()
        print('----Generate Random Walks----')
        rand_walk = n2v.simulate_walks()
        print('----Word2Vec----')
        n2v.learn_embeddings(rand_walk)
        print('--Completed--')
