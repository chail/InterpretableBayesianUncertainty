#!/usr/bin/env python3
# coding: utf-8

from toolbox.load_dataset import load_mnist

from torch.utils.data import DataLoader
from core_functions import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timeit import default_timer as timer
from collections import defaultdict
import pickle
import sys
import os

if len(sys.argv) != 3:
    print("Call this program like this:\n"
          "    ./mnist-mlp-train.py alpha run\n"
          "    e.g. ./mnist-mlp-train.py 0.5 1"
         )
    exit()

# extract command line arguments
alpha = float(sys.argv[1])
run = sys.argv[2]

# get dataset
train, validation, _ = load_mnist(flatten=True)

# constants
nb_train = train[0].shape[0]
nb_val = validation[0].shape[0]
input_dim = train[0].shape[1]
nb_classes = train[1].shape[1]

batch_size = 128
val_batch_size = nb_val
# nb_layers = 2
nb_units = 100
p = 0.5
wd = 1e-6

K_mc = 10

epochs = 30
use_cuda = False


# GPU configuation
if torch.cuda.is_available():
    device = torch.device("cuda:0" if use_cuda else "cpu")
else:
    device = torch.device("cpu")
    use_cuda = False


# set up datasets
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_dataset = MNISTDataset(train[0], train[1])
val_dataset = MNISTDataset(validation[0], validation[1])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, **kwargs)


# build network
net = Net(input_dim, nb_units, nb_classes, p)
net.to(device)

# optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=wd,
                      momentum=0.9, nesterov=True) # adds L2 regularization


# cross entropy loss if alpha = 0, otherwise alpha-divergence objective
loss_function = bbalpha_softmax_cross_entropy_with_mc_logits(alpha)

def train(model, device, train_loader, optimizer, ep, results):

    model.train()
    accum_n = 0.
    accum_total_loss = 0.
    accum_bb_loss = 0.
    accum_avg_acc = 0.
    accum_acc = 0.
    accum_ll = 0.
    tic = timer()

    for batch_idx, (data, labels) in enumerate(train_loader):

        data, labels = data.to(device), labels.to(device)
        data = data.float()
        labels = labels.float()
        labels_mc = to_categorical_mc(labels, K_mc)
        optimizer.zero_grad()

        outputs = [model(data) for _ in range(K_mc)]
        mc_outputs = torch.stack(outputs, dim=1)
        loss = loss_function(mc_outputs, labels_mc)
        acc = test_acc(mc_outputs, labels_mc)
        avg_acc, avg_ll = test_MC_dropout(mc_outputs, labels)
        l2 = wd * torch.sum(torch.stack([x.norm(2) for x in net.parameters()]))

        loss.backward()
        optimizer.step()

        n = data.size(0)
        nb_batches_train = nb_train / n

        accum_n += n
        accum_total_loss += (loss.item() + l2.item()) / nb_batches_train
        accum_bb_loss += loss.item() / nb_batches_train
        accum_acc += acc.item() / nb_batches_train
        accum_avg_acc += avg_acc.item() / nb_batches_train
        accum_ll += avg_ll.item() / nb_batches_train

    toc = timer()
    assert accum_n == nb_train, "Incomplete Training Epoch"
    results['train_N'].append(accum_n)
    results['train_total_loss'].append(accum_total_loss)
    results['train_bbalpha_loss'].append(accum_bb_loss)
    results['train_acc'].append(accum_acc)
    results['train_avg_acc'].append(accum_avg_acc)
    results['train_ll'].append(accum_ll)
    results['train_time'].append(toc-tic)
    print("Train Epoch: {}\tLoss: {:.3f}\tBB loss: {:.3f}\tAcc: {:.3f}\tAvg Acc: {:.3f}\tLL: {:.3f}\tTime: {:.3f}"
         .format(ep, accum_total_loss, accum_bb_loss, accum_acc, accum_avg_acc, accum_ll, toc-tic))

def validate(model, device, val_loader, ep, results):
    # model.eval() it still needs to be in train mode for stochastic output
    model.train()
    accum_n = 0.
    accum_total_loss = 0.
    accum_bb_loss = 0.
    accum_avg_acc = 0.
    accum_acc = 0.
    accum_ll = 0.
    tic = timer()
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            data = data.float()
            labels = labels.float()
            labels_mc = to_categorical_mc(labels, K_mc)
            outputs = [model(data) for _ in range(K_mc)]
            mc_outputs = torch.stack(outputs, dim=1)
            loss = loss_function(mc_outputs, labels_mc)
            acc = test_acc(mc_outputs, labels_mc)
            avg_acc, avg_ll = test_MC_dropout(mc_outputs, labels)
            l2 = wd * torch.sum(torch.stack([x.norm(2) for x in net.parameters()]))

            n = data.size(0)
            nb_batches_val = nb_val / n

            accum_n += n
            accum_total_loss += (loss.item() + l2.item()) / nb_batches_val
            accum_bb_loss += loss.item() / nb_batches_val
            accum_acc += acc.item() / nb_batches_val
            accum_avg_acc += avg_acc.item() / nb_batches_val
            accum_ll += avg_ll.item() / nb_batches_val

    # accumulate validation results
    toc = timer()
    assert accum_n == nb_val, "Incomplete Validation Run"
    results['val_N'].append(accum_n)
    results['val_total_loss'].append(accum_total_loss)
    results['val_bbalpha_loss'].append(accum_bb_loss)
    results['val_acc'].append(accum_acc)
    results['val_avg_acc'].append(accum_avg_acc)
    results['val_ll'].append(accum_ll)
    results['val_time'].append(toc-tic)
    print("Val Epoch: {}\tLoss: {:.3f}\tBB loss: {:.3f}\tAcc: {:.3f}\tAvg Acc: {:.3f}\tLL: {:.3f}\tTime: {:.3f}"
         .format(ep, accum_total_loss, accum_bb_loss, accum_acc, accum_avg_acc, accum_ll, toc-tic))

# Training loop
directory = os.path.join('saved_models', 'mnist-mlp-alpha{}-run{}'.format(alpha, run))
os.makedirs(directory, exist_ok=True)
results = defaultdict(list)
max_acc = 0.
max_acc_ep = 0
ep = 0

# while ep < max(2 * max_acc_ep, epochs):
while ep < epochs:
    train(net, device, train_loader, optimizer, ep, results)
    validate(net, device, val_loader, ep, results)

    val_avg_acc = results['val_avg_acc'][-1]
    if val_avg_acc > max_acc:
        max_acc = val_avg_acc
        max_acc_ep = ep
        print("Updating max_acc_ep: {}".format(max_acc_ep))

        # save the model
        tic = timer()
        torch.save(net.state_dict(), os.path.join(directory, 'model.pt'))
        toc = timer()
    print("Max_acc_ep: {}\t Max_acc: {:.3f}".format(max_acc_ep, max_acc))

    # save result after every epoch
    with open(os.path.join(directory, 'results.p'), 'wb') as f:
        pickle.dump(results, f)

    ep += 1

