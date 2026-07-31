#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_session_representation(self, hidden, mask):
        if hidden.ndim != 3:
            raise ValueError("SR-GNN hidden states must have shape [batch, sequence, hidden].")
        if mask.ndim != 2:
            raise ValueError("SR-GNN session masks must have shape [batch, sequence].")
        if hidden.shape[:2] != mask.shape:
            raise ValueError(
                "SR-GNN hidden states and session masks must share batch and sequence dimensions."
            )

        mask = mask.to(device=hidden.device)
        last_positions = torch.sum(mask, dim=1, dtype=torch.long) - 1
        batch_positions = torch.arange(
            hidden.shape[0], dtype=torch.long, device=hidden.device
        )
        ht = hidden[batch_positions, last_positions]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        return a

    def compute_scores(self, hidden, mask):
        a = self.compute_session_representation(hidden, mask)
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def validate_session_mask_array(mask):
    mask_array = np.asarray(mask, dtype=np.int64)
    if mask_array.ndim != 2:
        raise ValueError("SR-GNN session masks must have shape [batch, sequence].")
    if mask_array.shape[0] == 0 or mask_array.shape[1] == 0:
        raise ValueError("SR-GNN session masks must contain at least one non-empty session.")
    if np.any((mask_array != 0) & (mask_array != 1)):
        raise ValueError("SR-GNN session masks must contain only 0 and 1 values.")

    session_lengths = mask_array.sum(axis=1, dtype=np.int64)
    if np.any(session_lengths <= 0):
        raise ValueError("SR-GNN received an empty session in a training or evaluation batch.")
    if np.any(session_lengths > mask_array.shape[1]):
        raise ValueError("SR-GNN session length exceeds the padded sequence width.")
    return mask_array


def validate_srgnn_batch_arrays(
    *,
    alias_inputs,
    adjacency,
    items,
    mask,
    targets,
    n_node,
    sample_indices=None,
):
    alias_array = np.asarray(alias_inputs, dtype=np.int64)
    adjacency_array = np.asarray(adjacency, dtype=np.float32)
    items_array = np.asarray(items, dtype=np.int64)
    mask_array = validate_session_mask_array(mask)
    targets_array = np.asarray(targets, dtype=np.int64)
    indices_array = (
        None if sample_indices is None else np.asarray(sample_indices, dtype=np.int64)
    )

    if items_array.ndim != 2 or items_array.shape[0] == 0 or items_array.shape[1] == 0:
        raise ValueError("SR-GNN batch items must have a non-empty 2D shape.")
    if alias_array.shape != mask_array.shape:
        raise ValueError(
            "SR-GNN alias inputs and masks must share batch and sequence dimensions."
        )
    if alias_array.shape[0] != items_array.shape[0]:
        raise ValueError("SR-GNN alias inputs and items must share the batch dimension.")
    if targets_array.ndim != 1 or targets_array.shape[0] != items_array.shape[0]:
        raise ValueError("SR-GNN targets must contain exactly one label per batch row.")
    if adjacency_array.shape != (
        items_array.shape[0],
        items_array.shape[1],
        2 * items_array.shape[1],
    ):
        raise ValueError(
            "SR-GNN adjacency must have shape [batch, nodes, 2 * nodes]."
        )

    def _location(row_index):
        if indices_array is None or row_index >= len(indices_array):
            return f"batch_row={row_index}"
        return f"batch_row={row_index} sample_index={int(indices_array[row_index])}"

    bad_items = np.argwhere((items_array < 0) | (items_array >= int(n_node)))
    if bad_items.size:
        row_index, column_index = map(int, bad_items[0])
        raise ValueError(
            "SR-GNN item id is outside the embedding range: "
            f"{_location(row_index)} column={column_index} "
            f"value={int(items_array[row_index, column_index])} "
            f"expected=0..{int(n_node) - 1}."
        )

    bad_alias = np.argwhere(
        (alias_array < 0) | (alias_array >= int(items_array.shape[1]))
    )
    if bad_alias.size:
        row_index, column_index = map(int, bad_alias[0])
        raise ValueError(
            "SR-GNN alias index is outside the batch node range: "
            f"{_location(row_index)} column={column_index} "
            f"value={int(alias_array[row_index, column_index])} "
            f"expected=0..{int(items_array.shape[1]) - 1}."
        )

    bad_targets = np.argwhere(
        (targets_array < 1) | (targets_array >= int(n_node))
    )
    if bad_targets.size:
        row_index = int(bad_targets[0, 0])
        raise ValueError(
            "SR-GNN target id is outside the score range: "
            f"{_location(row_index)} value={int(targets_array[row_index])} "
            f"expected=1..{int(n_node) - 1}."
        )

    return (
        alias_array,
        adjacency_array,
        items_array,
        mask_array,
        targets_array,
    )


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs, A, items, mask, targets = validate_srgnn_batch_arrays(
        alias_inputs=alias_inputs,
        adjacency=A,
        items=items,
        mask=mask,
        targets=targets,
        n_node=model.n_node,
        sample_indices=i,
    )
    alias_inputs = trans_to_cuda(torch.from_numpy(alias_inputs))
    items = trans_to_cuda(torch.from_numpy(items))
    A = trans_to_cuda(torch.from_numpy(A))
    mask = trans_to_cuda(torch.from_numpy(mask))
    hidden = model(items, A)
    seq_hidden = torch.stack(
        [hidden[row_index][alias_inputs[row_index]] for row_index in range(len(alias_inputs))]
    )
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data, log_batches=True):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        try:
            model.optimizer.zero_grad()
            targets, scores = forward(model, i, train_data)
            targets = trans_to_cuda(torch.from_numpy(targets))
            loss = model.loss_function(scores, targets - 1)
            loss.backward()
            model.optimizer.step()
            # Accumulate a detached host scalar so the epoch total does not retain
            # a chain of thousands of completed autograd graphs on large datasets.
            total_loss += float(loss.detach().item())
        except Exception:
            first_index = int(i[0]) if len(i) else None
            last_index = int(i[-1]) if len(i) else None
            print(
                "[srgnn] training batch failed: "
                f"batch={int(j) + 1}/{len(slices)} "
                f"sample_indices={first_index}..{last_index}"
            )
            raise
        if log_batches and j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    avg_loss = total_loss / max(1, len(slices))
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    with torch.no_grad():
        for i in slices:
            targets, scores = forward(model, i, test_data)
            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, mask in zip(sub_scores, targets, test_data.mask):
                hit.append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr, avg_loss
