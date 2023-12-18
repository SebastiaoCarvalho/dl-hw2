#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

import utils

class CNN(nn.Module):
    
    def __init__(self, width, height, n_classes, dropout_prob, no_maxpool=False):
        super(CNN, self).__init__()
        self.no_maxpool = no_maxpool
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_prob)
        if not no_maxpool:
            # Implementation for Q2.1
            self.conv1 = nn.Conv2d(1, 8, (3, 3), padding=3-1)
            self.pool1 = nn.MaxPool2d((2, 2))
            width = width // 2
            height = height // 2
            self.conv2 = nn.Conv2d(8, 16, (3, 3))
            width = width - 2
            height = height - 2
            self.pool2 = nn.MaxPool2d((2, 2))
            width = width // 2
            height = height // 2
            self.fc1 = nn.Linear(1, 320) # chanfe input size in forward method ????
            self.fc2 = nn.Linear(320, 120)
            self.fc3 = nn.Linear(120, n_classes)
        else:
            # Implementation for Q2.2
            raise NotImplementedError

    def forward(self, x):
        # input should be of shape [b, c, w, h]
        # conv and relu layers
        x = self.conv1(x)
        x = self.relu(x)
        # max-pool layer if using it
        if not self.no_maxpool:
            x = self.pool1(x)
        
        # conv and relu layers
        x = self.conv2(x)
        x = self.relu(x)

        # max-pool layer if using it
        if not self.no_maxpool:
            x = self.pool2(x)
        
        # prep for fully connected layer + relu
        x = x.view(x.shape[0], -1)
        x = nn.Linear(x.shape[1], 320)(x)
        x = self.relu(x)
        
        # drop out
        x = self.drop(x)

        # second fully connected layer + relu
        x = self.fc2(x)
        x = self.relu(x)
        
        # last fully connected layer
        x = self.fc3(x)
        
        return F.log_softmax(x,dim=1)

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    optimizer.zero_grad()
    features = X.shape[1]
    X.resize_(X.shape[0], 1, int(np.sqrt(features)), int(np.sqrt(features)))
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def get_number_trainable_params(model):
    ## TO IMPLEMENT - REPLACE return 0
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=15, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.7)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')
    
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_oct_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]

    # initialize the model
    model = CNN(0, 0, n_classes, opt.dropout, no_maxpool=opt.no_maxpool)
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )
    
    # get a loss criterion
    criterion = nn.NLLLoss()
    
    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    config = "{}-{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer, opt.no_maxpool)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    
    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()
