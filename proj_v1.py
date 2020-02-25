import argparse
import logging
import os
import time
import tqdm

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torch.autograd import Variable

import data_loader
from model_v1 import resnet50

"""
    define a train process
    args: train dataloader
"""
def train(dataloader, params_training):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = resnet50().to(device)
    model = torch.nn.DataParallel(model)

    model.train()

    # crossentropy and adam optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params_training["learning_rate"])

    loss_list = []
    training_step = len(dataloader)

    for epoch in range(params_training["num_epoch"]):
        print('epoch {}'.format(epoch))
        # with tqdm(total=157) as t:
        for i, (train_batch_orig, labels_batch) in enumerate(dataloader):
            train_batch_orig, labels_batch = Variable(train_batch_orig), Variable(labels_batch)
            train_batch = train_batch_orig / 255.0

            out_batch = model(train_batch)
            loss = criterion(out_batch, labels_batch)
            loss_list.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 2 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, params_training["num_epoch"], i + 1, training_step, loss.item()))

                # t.set_postfix(loss='{:05.3f}'.format(loss.item()))
                # t.update()


if __name__ == '__main__':
    start = time.time()

    params = {
        "batch_size": 64,
        "num_workers": 4,
        "shuffle": True
    }

    params_training = {
        "learning_rate": 1e-1,
        "dropout_rate": 0.8,
        "num_epoch": 20
    }

    path = 'data'
    train_data = os.path.join(path, 'train_mix')
    test_data = os.path.join(path, 'test_mix')
    evl_data = os.path.join(path, 'val_mix')

    dataloaders = data_loader.fetch_dataloader(['train', 'val', 'test'], path, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    test_dl = dataloaders['test']

    train(train_dl, params_training)

    end = time.time()
    print('Training Time: ', end-start)