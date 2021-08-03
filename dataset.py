

import os, sys, hashlib
import abc
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import torch
import sys
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import dgl
import cplex
import networkx as nx
# import visdom
import matplotlib.pyplot as plt
import random
from ultis import graphPartition


def collate_fn1(samples):
    og_list = [i[0] for i in samples]
    sep_list = [i[1] for i in samples]
    # return dgl.batch(og_list), dgl.batch(og_list)
    return dgl.batch(og_list), dgl.batch(sep_list)


def collate_fn2(samples):
    og_list = [i[0] for i in samples]
    return dgl.batch(og_list), dgl.batch(og_list)
    # return dgl.batch(og_list), dgl.batch(sep_list)

class DataSet(DGLDataset):

    def __init__(self, name, seed, device, with_sep, url=None, raw_dir=None, save_dir=None,
                     hash_key=(), force_reload=False, verbose=False):
        
        # if train:
        #     self._name = agrs.train_dataset_name
        # else:
        #     self._name = agrs.test_dataset_name

        self.with_sep = with_sep


        if self.with_sep:
            self.collate_fn = collate_fn1
        else:
            self.collate_fn = collate_fn2

        self._name = name

        self.kwargs = {'pin_memory': True}

        self.seed = seed
        self.device = device


        self._url = url
        self._force_reload = force_reload
        self._verbose = verbose
        self._hash_key = hash_key
        self._hash_func = hashlib.sha1()
        self._raw_dir = "data/"
        self._save_dir = self._raw_dir
        # self._hash = self._get_hash()

        # # if no dir is provided, the default dgl download dir is used.
        # if agrs.raw_dir is None:
        #     self._raw_dir = "data/"
        #     # self._raw_dir = get_download_dir()
        # else:
        #     self._raw_dir = agrs.raw_dir

        # if agrs.save_dir is None:
        #     self._save_dir = self._raw_dir
        # else:
        #     self._save_dir = agrs.save_dir

        # self._save_dir = agrs.save_dir
        self._load()
        # self._optimal_dict = None
        # self._optimal_list = None



    #In MaxDataset we generate data instead of downloading
    def download(self):

        raise NotImplementedError("Please generate data first")


        # print(self._name)

        # self._graphs, self._optimal_dict, self.con_list, self.e2n_list, self.n2e_list  = data_generator.generate(self._name, self._raw_dir, "MaxCut")
        # print("showing: self._optimal_dict", self._optimal_dict)
        # self._optimal_list = self._optimal_dict["cplex"].numpy()
        # print("showing: self._optimal_list", self._optimal_list)
        # print("Len = ", len(self._optimal_list))

    def process(self):
        #nothing to do here
        pass

    def print_info(self):
        e1, e2 = self._graphs[0].edges()
        e1 = e1.numpy()
        e2 = e2.numpy()
        if self.with_sep:
            self.gen_sep_g()
        else:
            self.sep_graphs = self._graphs

        print('Finished data loading.')
        print('Show a example of graphs')
        print('  NumNodes: {}'.format(self._graphs[0].number_of_nodes()))
        print('  NumEdges: {}'.format(self._graphs[0].number_of_edges()))
        print('  Edges: ', e1)
        print("         ", e2)
        print("  Edges number * 2 = ", len(e1))




    def has_cache(self):
        graph_path = self._name
        # graph_path = os.path.join(self._name, '_dgl_graph.bin')
        print(graph_path)
        if os.path.exists(graph_path):
            print("data already exists")
        else:
            print("data do not exist")

        return os.path.exists(graph_path)

    def save(self):
        graph_path = self._name
        # graph_path = os.path.join(self._name, '_dgl_graph.bin')

        save_graphs(graph_path, self._graphs, None)

    def gen_sep_g(self):

        self.sep_graphs = []
        graphs = self._graphs
        for i in range(len(graphs)):
            # print(i)

            # graphPartition(graphs[4], int(graphs[4].num_nodes()/10))
            # exit()
            nxg = dgl.to_networkx(graphs[i]).to_undirected()
            

            if nx.is_connected(nxg):

                clusters =  graphPartition(graphs[i], int(graphs[i].num_nodes()/10))

                tmp = [ dgl.node_subgraph(graphs[i], cluster) for cluster in clusters   ]
                qaq = dgl.batch(tmp)
                # print(qaq)
                qaq.ndata.pop('_ID')
                qaq.edata.pop('_ID')
                self.sep_graphs.append(qaq)
            else:
                self.sep_graphs.append(graphs[i])


    def load(self):
        # print("?????????????????")
        graph_path = self._name
        # graph_path = os.path.join(self._name, '_dgl_graph.bin')

        graphs, _ = load_graphs(graph_path)
        self._graphs = graphs



        # print(self.sep_graphs)
        # exit()
        # for i in range(len(graphs)):
            # graphs[i] =  dgl.add_nodes(graphs[i], 1)

        # print(graphs)
        # exit()
        



    def __len__(self):
        """Return number of samples in this dataset."""
        return len(self._graphs)

    def __getitem__(self, item):
        """Get the item^th sample.

        Parameters
        ---------
        item : int
            The sample index.

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node features and node labels.

            - ``ndata['feat']``: node features
            - ``ndata['label']``: node labels
        """
        # print(type(self._graphs), type(self._optimal_list))
        return self._graphs[item], self.sep_graphs[item]


    def getloader(self, batch_size):


        return DataLoader(
            self, batch_size = batch_size, collate_fn = self.collate_fn, **self.kwargs)
