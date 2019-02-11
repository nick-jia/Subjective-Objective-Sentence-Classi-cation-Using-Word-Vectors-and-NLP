import torch
import torch.optim as optim
import json

import torchtext
import spacy

import argparse
import os

import numpy as np
from time import time

from models import *


def evaluate(model, iterator, loss_fnc):
    total_val_loss = 0.0
    total_val_err = 0.0
    total_epoch = 0
    i = 0

    for data in iterator:
        (x, x_lengths), y = data.text, data.label
        outputs = model(x, lengths=x_lengths).squeeze()
        loss = loss_fnc(outputs, y.float())

        x = outputs > 0.5
        corr = x.long() != y
        total_val_err += int(corr.sum())
        total_val_loss += loss.item()
        total_epoch += len(y)
        i += 1

    return float(total_val_err) / total_epoch, float(total_val_loss) / (i + 1)


def main(args):
    filter_sizes = [2, 4]
    save = True

    text_field = torchtext.data.Field(sequential=True, tokenize='spacy', include_lengths=True)
    label_field = torchtext.data.Field(sequential=False, use_vocab=False)

    train, val, test = torchtext.data.TabularDataset.splits(path='./data/', train='train.tsv',
                                                            validation='validation.tsv', test='test.tsv',
                                                            skip_header=True, format='TSV',
                                                            fields=[('text', text_field), ('label', label_field)])

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits((train, val, test),
                                                                           batch_sizes=(args.batch_size,
                                                                                        args.batch_size,
                                                                                        args.batch_size),
                                                                           sort_key=lambda x: len(x.text),
                                                                           sort_within_batch=True, repeat=False)

    # train_iter, val_iter, test_iter = torchtext.data.Iterator.splits((train, val, test), batch_sizes=(args.batch_size,
    #                                                                                                   args.batch_size,
    #                                                                                                   args.batch_size),
    #                                                                  sort_key=lambda x: len(x.text),
    #                                                                  sort_within_batch=True, repeat=False)

    text_field.build_vocab(train)

    glove_model = torchtext.vocab.GloVe(name='6B', dim=100)
    text_field.vocab.load_vectors(glove_model)

    if args.model == 'baseline':
        model = Baseline(args.emb_dim,text_field.vocab)

    elif args.model == 'rnn':
        model = RNN(args.emb_dim,text_field.vocab,args.rnn_hidden_dim)

    else:
        model = CNN(args.emb_dim, text_field.vocab, args.num_filt, filter_sizes)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
    loss_fnc = torch.nn.BCEWithLogitsLoss()

    train_err = np.ones(args.epochs)
    train_loss = np.ones(args.epochs)
    val_err = np.ones(args.epochs)
    val_loss = np.ones(args.epochs)

    start_time = time()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        i = 0

        for data in train_iter:
            (x, x_lengths), y = data.text, data.label
            optimizer.zero_grad()
            outputs = model(x, lengths=x_lengths).squeeze()
            loss = loss_fnc(outputs, y.float())
            loss.backward()
            optimizer.step()

            x = outputs > 0.5
            corr = x.long() != y
            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(y)
            i += 1

        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i + 1)
        val_err[epoch], val_loss[epoch] = evaluate(model, val_iter, loss_fnc)
        if val_err[epoch] == min(val_err):
            best_model, best_epoch, best_accuracy = model, epoch, 1 - val_err[epoch]

        print("Epoch {}: Train err: {}, Train loss: {} | Validation err: {}, Validation loss: {}"
              .format(epoch + 1, train_err[epoch], train_loss[epoch], val_err[epoch], val_loss[epoch]))

    print('Finished Training, best accuracy is {} at {} epoch.'.format(best_accuracy, best_epoch + 1))
    test_err, test_loss = evaluate(model, test_iter, loss_fnc)
    print('Test accuracy is {}, test loss is {}'.format(1 - test_err, test_loss))
    end_time = time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    if save:
        torch.save(model, 'model_{}.pt'.format(args.model))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)
