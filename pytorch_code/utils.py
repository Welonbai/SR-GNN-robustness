#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import networkx as nx
import numpy as np


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs = np.asarray(self.inputs[i], dtype=np.int64)
        targets = np.asarray(self.targets[i], dtype=np.int64)
        # Item id 0 is SR-GNN's padding value. Rebuild each batch mask from the
        # authoritative padded inputs instead of trusting the long-lived cached
        # mask array, which may be corrupted independently during large runs.
        mask = np.not_equal(inputs, 0).astype(np.int64, copy=False)
        if inputs.ndim != 2 or inputs.shape[0] == 0 or inputs.shape[1] == 0:
            raise ValueError("SR-GNN batches must contain a non-empty 2D input array.")

        # Resolve the sorted unique nodes once per session. The legacy code ran
        # np.unique twice and accumulated several nested Python lists per batch;
        # preallocating the output arrays reduces allocator churn while keeping
        # the exact node order and graph normalization unchanged.
        nodes = [np.unique(u_input) for u_input in inputs]
        max_n_node = max(len(node) for node in nodes)
        batch_size, sequence_width = inputs.shape
        items = np.zeros((batch_size, max_n_node), dtype=np.int64)
        alias_inputs = np.empty((batch_size, sequence_width), dtype=np.int64)
        A = np.zeros((batch_size, max_n_node, 2 * max_n_node), dtype=np.float64)

        for row_index, (u_input, node) in enumerate(zip(inputs, nodes)):
            items[row_index, : len(node)] = node
            node_positions = {
                int(item): int(position) for position, item in enumerate(node)
            }
            alias_inputs[row_index] = [
                node_positions[int(item)] for item in u_input
            ]

            u_A = np.zeros((max_n_node, max_n_node), dtype=np.float64)
            for position in range(len(u_input) - 1):
                if u_input[position + 1] == 0:
                    break
                u = node_positions[int(u_input[position])]
                v = node_positions[int(u_input[position + 1])]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            A[row_index] = np.concatenate([u_A_in, u_A_out]).transpose()
        return alias_inputs, A, items, mask, targets
