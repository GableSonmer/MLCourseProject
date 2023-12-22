import datetime
import os
import re

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch

from exp import Exp
from model import LSTMModel, TransformerModel
from oil_dataset import OilDataset
import matplotlib.pyplot as plt
import argparse

COLUMN_NAMES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']


def parse_args():
    parser = argparse.ArgumentParser(description='Oil Temperature Time Series Forecasting')
    # input size and output size, can be removed
    # parser.add_argument('--input_size', type=int, default=7, help='input size')
    # parser.add_argument('--output_size', type=int, default=7, help='output size')

    # visualization setting, plot result or not
    parser.add_argument('--plot', type=bool, default=False, help='plot result or not')

    # training setting, learning rate, epochs, batch size, patience
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    # length setting
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token index of input sequence')
    parser.add_argument('--pred_len', type=int, default=336, help='predict sequence length')

    # inverse setting
    parser.add_argument('--inverse', type=bool, default=True, help='inverse data or not')

    # model setting
    parser.add_argument('--model', type=str, default='lstm', help='model name', choices=['lstm', 'transformer'])

    # optimizer setting
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer', choices=['adam', 'sgd', 'rmsprop'])

    # criterion setting
    parser.add_argument('--criterion', type=str, default='mse', help='criterion', choices=['mse', 'mae'])

    # gpu setting
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')

    # number of workers setting
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')

    # iteration setting, default is 5
    parser.add_argument('--iteration', type=int, default=5, help='number of iterations for experiment')

    # lr adjust setting
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate type',
                        choices=['type1', 'type2'])
    return parser.parse_args()


def main(args):
    args.use_gpu = torch.cuda.is_available()
    print('Experiment setting:')
    print(args)
    for i in range(args.iteration):
        if i > 0:
            print('=' * 50)
        print(f'Iteration {i + 1}')
        # {model}_sqln{seq_len}_ll{label_len}_pl{pred_len}_lr{lr}_ep{epochs}_bs{batch_size}_op{optimizer}_cr{criterion}
        setting = '{}_sqln{}_ll{}_pl{}_lr{}_ep{}_bs{}_op{}_cr{}'.format(
            args.model,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.lr,
            args.epochs,
            args.batch_size,
            args.optimizer,
            args.criterion
        )

        exp = Exp(args)

        # train
        print(f'Start training : {setting} ...........')
        exp.train(setting)

        # test
        print(f'Start testing : {setting} ...........')
        exp.test(setting)

        torch.cuda.empty_cache()


def plot():
    folders = os.listdir('results')
    folders = sorted(filter(lambda x: x.startswith('lstm') or x.startswith('transformer'), folders))
    length = len(folders)
    folder = ''
    while True:
        try:
            for i in range(length):
                print(f'#{i}: {folders[i]}')
            # index = input('Select the index of the folder you want to plot: ')
            index = 0
            folder = folders[int(index)]
            break
        except:
            print('Invalid input, try again')
            continue

    # extract predict length from folder name use regex
    path = os.path.join('results', folder)
    pred_len = re.findall(r'pl(\d+)_', folder)[0]

    # load data
    pred = np.load(os.path.join(path, 'pred.npy'))
    true = np.load(os.path.join(path, 'true.npy'))

    assert pred.shape == true.shape
    print('Shape', pred.shape)

    idx = 0
    pred = pred[idx]
    true = true[idx]

    plt.figure(figsize=(10, 30))
    for i in range(7):
        plt.subplot(7, 1, i + 1)
        x1 = pred[:, i]
        x2 = true[:, i]
        plt.title(f'Feature {i + 1}')
        plt.plot(x1, label='pred')
        plt.plot(x2, label='true')
        plt.legend()

    plt.savefig('out.png')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    args.plot = True

    if args.plot:
        plot()
    else:
        main(args)
