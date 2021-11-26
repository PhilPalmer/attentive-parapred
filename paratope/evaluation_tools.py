"""
Helper functions for the evaluation suite.
"""
import numpy as np
import torch
torch.set_printoptions(threshold=50000)
from torch import index_select

from constants import *

def sort_batch(cdrs, masks, lengths, lbls):
    """
    Sort antibody amino acid sequences by length -- for RNN
    :param cdrs:
    :param masks:
    :param lengths:
    :param lbls:
    :return: sorted input
    """
    order = np.argsort(lengths)
    order = order.tolist()
    order.reverse()
    lengths.sort(reverse=True)
    index = torch.LongTensor(order)
    if use_cuda:
        index = index.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    lbls = torch.index_select(lbls, 0, index)
    masks = torch.index_select(masks, 0, index)
    return cdrs, masks, lengths, lbls

def sort_ag_batch(cdrs, masks, lengths, lbls, ag, ag_masks, dist, delta_gs):
    order = np.argsort(lengths)
    order = order.tolist()
    order.reverse()

    lengths.sort(reverse=True)

    index = torch.LongTensor(order)
    if use_cuda:
        index = index.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    lbls = torch.index_select(lbls, 0, index)
    masks = torch.index_select(masks, 0, index)

    ag = torch.index_select(ag, 0, index)
    ag_masks = torch.index_select(ag_masks, 0, index)

    dist= torch.index_select(dist, 0, index)
    delta_gs = torch.index_select(delta_gs, 0, index)

    return cdrs, masks, lengths, lbls, ag, ag_masks, dist, delta_gs

def sort_probs(probs):
    print("probs", probs)
    probs.sort()
    return probs

def vis_sort_batch(cdrs, masks, lengths, lbls):
    order = np.argsort(lengths)
    order = order.tolist()
    order.reverse()
    lengths.sort(reverse=True)
    index = torch.LongTensor(order)
    #index = torch.LongTensor(order)

    if use_cuda:
        index = index.cuda()
        cdrs = cdrs.cuda()
        lbls = lbls.cuda()
        masks = masks.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    lbls = torch.index_select(lbls, 0, index)
    masks = torch.index_select(masks, 0, index)
    return cdrs, masks, lengths, lbls, index

def sort_batch_without_labels(cdrs, masks, lengths):
    order = np.argsort(lengths)
    order = order.tolist()
    order.reverse()
    lengths.sort(reverse=True)
    index = torch.LongTensor(order)
    #index = torch.LongTensor(order)

    if use_cuda:
        index = index.cuda()
        cdrs = cdrs.cuda()
        masks = masks.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    masks = torch.index_select(masks, 0, index)
    return cdrs, masks, lengths, index

def ag_vis_sort_batch(cdrs, masks, lengths, lbls, ag, ag_masks):
    order = np.argsort(lengths)
    order = order.tolist()
    order.reverse()
    lengths.sort(reverse=True)
    index = torch.LongTensor(order)
    #index = torch.LongTensor(order)

    if use_cuda:
        index = index.cuda()
        cdrs = cdrs.cuda()
        lbls = lbls.cuda()
        masks = masks.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    lbls = torch.index_select(lbls, 0, index)
    masks = torch.index_select(masks, 0, index)
    ag = torch.index_select(ag, 0, index)
    ag_masks = torch.index_select(ag_masks, 0, index)
    return cdrs, masks, lengths, lbls, index, ag, ag_masks

def permute_training_ag_data(cdrs, masks, lengths, lbls, ag, ag_masks, ag_lengths, dist, delta_gs):
    index = torch.randperm(cdrs.shape[0])
    if use_cuda:
        index = index.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    lbls = torch.index_select(lbls, 0, index)
    masks = torch.index_select(masks, 0, index)
    lengths = [lengths[i] for i in index]

    ag = torch.index_select(ag, 0, index)
    ag_masks = torch.index_select(ag_masks, 0, index)
    ag_lengths = [ag_lengths[i] for i in index]

    dist= torch.index_select(dist, 0, index)
    delta_gs = torch.index_select(delta_gs, 0, index)

    return cdrs, masks, lengths, lbls, ag, ag_masks, ag_lengths, dist, delta_gs

def permute_training_data(cdrs, masks, lengths, lbls):
    """
    Shuffle training data
    :param cdrs:
    :param masks:
    :param lengths:
    :param lbls:
    :return:
    """
    index = torch.randperm(cdrs.shape[0])
    if use_cuda:
        index = index.cuda()

    cdrs = torch.index_select(cdrs, 0, index)
    lbls = torch.index_select(lbls, 0, index)
    masks = torch.index_select(masks, 0, index)
    lengths = [lengths[i] for i in index]

    return cdrs, masks, lengths, lbls

def flatten_with_lengths(matrix, lengths):
    seqs = []
    for i, example in enumerate(matrix):
        seq = example[:lengths[i]]
        seqs.append(seq)
    return np.concatenate(seqs)
