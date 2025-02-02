"""
Training and testing AG-Fast-Parapred
"""
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import index_select
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
import time

from constants import *
from evaluation_tools import *
from cross_self_model import *

def xself_run(cdrs_train, lbls_train, masks_train, lengths_train,
               ag_train, ag_masks_train, ag_lengths_train, dist_mat_train,
               weights_template, weights_template_number,
               cdrs_test, lbls_test, masks_test, lengths_test,
               ag_test, ag_masks_test, ag_lengths_test, dist_test):
    """

    :param cdrs_train: antibody amino acids used for training
    :param lbls_train: ground truth values for antibody amino acids used for training
    :param masks_train: amino acids' masks
    :param lengths_train: amino acids' lengths
    :param ag_train: antigen amino acids used for training
    :param ag_masks_train: antigen amino acids' masks
    :param ag_lengths_train: antigen amino acids' lengths
    :param dist_mat_train:
    :param weights_template: template for printing weights
    :param weights_template_number: which file to print weights to
    :param cdrs_test: antibody amino acids used for testing
    :param lbls_test:
    :param masks_test:
    :param lengths_test:
    :param ag_test:
    :param ag_masks_test:
    :param ag_lengths_test:
    :param dist_test:
    :return:
    """

    print("dilated run", file=print_file)
    model = XSelf()

    ignored_params = list(map(id, [model.conv1.weight, model.conv2.weight, model.conv3.weight,
                                   model.agconv1.weight, model.agconv2.weight, model.agconv3.weight,
                                   model.aconv1.weight, model.aconv2.weight]))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())

    optimizer = optim.Adam([
        {'params': base_params},
        {'params': model.conv1.weight, 'weight_decay': 0.01},
        {'params': model.conv2.weight, 'weight_decay': 0.01},
        {'params': model.conv3.weight, 'weight_decay': 0.01},
        {'params': model.agconv1.weight, 'weight_decay': 0.01},
        {'params': model.agconv2.weight, 'weight_decay': 0.01},
        {'params': model.agconv3.weight, 'weight_decay': 0.01},
        {'params': model.aconv1.weight, 'weight_decay': 0.01},
        {'params': model.aconv2.weight, 'weight_decay': 0.01}
    ], lr=0.01)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    total_input = cdrs_train
    total_lbls = lbls_train
    total_masks = masks_train
    total_lengths = lengths_train
    total_dist_train = dist_mat_train

    total_ag_input = ag_train
    total_ag_masks = ag_masks_train
    total_ag_lengths = ag_lengths_train

    if use_cuda:
        print("using cuda")
        model.cuda()
        total_input = total_input.cuda()
        total_lbls = total_lbls.cuda()
        total_masks = total_masks.cuda()
        cdrs_test = cdrs_test.cuda()
        lbls_test = lbls_test.cuda()
        masks_test = masks_test.cuda()

        total_ag_input = total_ag_input.cuda()
        total_ag_masks = total_ag_masks.cuda()
        ag_test = ag_test.cuda()
        ag_masks_test = ag_masks_test.cuda()

        total_dist_train = total_dist_train.cuda()
        dist_test = dist_test.cuda()

    times = []

    for epoch in range(epochs):  # training iterations
        model.train(True)
        scheduler.step()
        epoch_loss = 0

        batches_done=0

        total_input, total_masks, total_lengths, total_lbls,\
        total_ag_input, total_ag_masks, total_ag_lengths, total_dist_train = \
            permute_training_ag_data(total_input, total_masks, total_lengths, total_lbls,
                                     total_ag_input, total_ag_masks, total_ag_lengths, total_dist_train)

        total_time = 0

        for j in range(0, cdrs_train.shape[0], batch_size):
            batches_done +=1
            interval = [x for x in range(j, min(cdrs_train.shape[0], j + batch_size))]
            interval = torch.LongTensor(interval)
            if use_cuda:
                interval = interval.cuda()

            input = index_select(total_input, 0, interval)
            lbls = index_select(total_lbls, 0, interval)
            masks = index_select(total_masks, 0, interval)
            lengths = total_lengths[j:j + batch_size]

            ag_input = index_select(total_ag_input, 0, interval)
            ag_masks = index_select(total_ag_masks, 0, interval)

            dist = index_select(total_dist_train, 0, interval)

            input, masks, lengths, lbls, ag, ag_masks, dist = \
                sort_ag_batch(input, masks, list(lengths), lbls, ag_input, ag_masks, dist)

            output, _ = model(input, masks, ag_input, ag_masks, dist)

            loss_weights = (lbls * 1.5 + 1) * masks
            max_val = (-output).clamp(min=0)
            loss = loss_weights * \
                   (output - output * lbls + max_val + ((-max_val).exp() + (-output - max_val).exp()).log())
            masks_added = masks.sum()
            loss = loss.sum() / masks_added
            print("Epoch %d - Batch %d has loss %d " % (epoch, j, loss.data), file=monitoring_file)
            epoch_loss +=loss
            model.zero_grad()

            start_time =time.time()

            loss.backward()
            optimizer.step()

            total_time += time.time() - start_time
            #print("Total time", total_time)

        #print("Epoch %d - loss is %f : " % (epoch, epoch_loss.data[0]/batches_done))
        #print("--- %s seconds ---" % (total_time))
        times.append(total_time)

        model.eval()

        cdrs_test2, masks_test2, lengths_test2, lbls_test2, ag_test2, ag_masks_test2, dist_test2 = \
            sort_ag_batch(cdrs_test, masks_test, list(lengths_test), lbls_test, ag_test, ag_masks_test, dist_test)


        probs_test2, _= model(cdrs_test2, masks_test2, ag_test2, ag_masks_test2, dist_test2)


        sigmoid = nn.Sigmoid()
        probs_test2 = sigmoid(probs_test2)

        probs_test2 = probs_test2.data.cpu().numpy().astype('float32')
        lbls_test2 = lbls_test2.data.cpu().numpy().astype('int32')

        probs_test2 = flatten_with_lengths(probs_test2, lengths_test2)
        lbls_test2 = flatten_with_lengths(lbls_test2, lengths_test2)

        print("Roc", roc_auc_score(lbls_test2, probs_test2))

    print("Saving")

    torch.save(model.state_dict(), weights_template.format(weights_template_number))

    times_mean = np.mean(times)
    times_std = 2 * np.std(times)

    #print("Time mean", times_mean)
    #print("Time std", times_std)

    model.eval()

    cdrs_test, masks_test, lengths_test, lbls_test, ag_test, ag_masks_test, dist_test = \
        sort_ag_batch(cdrs_test, masks_test, list(lengths_test), lbls_test, ag_test, ag_masks_test, dist_test)

    probs_test, _ = model(cdrs_test, masks_test, ag_test, ag_masks_test, dist_test)


    sigmoid = nn.Sigmoid()
    probs_test = sigmoid(probs_test)

    #print("probs", probs_test, file=track_f)

    probs_test1 = probs_test.data.cpu().numpy().astype('float32')
    lbls_test1 = lbls_test.data.cpu().numpy().astype('int32')

    probs_test1 = flatten_with_lengths(probs_test1, list(lengths_test))
    lbls_test1 = flatten_with_lengths(lbls_test1, list(lengths_test))

    print("Roc", roc_auc_score(lbls_test1, probs_test1))

    return probs_test, lbls_test, probs_test1, lbls_test1  # get them in kfold, append, concatenate do roc on them
