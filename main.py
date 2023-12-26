import os
import re

import numpy as np
import pandas as pd
import torch

from exp import Exp
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

    # draw how many samples
    parser.add_argument('--plot_samples', type=int, default=10, help='plot how many samples')
    return parser.parse_args()


def main(args):
    args.use_gpu = torch.cuda.is_available()
    print('Experiment setting:')
    print(args)
    metrics = []
    folder_path = ''
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
        folder_path, mse, mae = exp.test(setting)
        metrics.append([i, mse, mae])
        torch.cuda.empty_cache()

    # save metrics
    df = pd.DataFrame(metrics, columns=['iteration', 'mse', 'mae'])
    df.to_csv(os.path.join('results', folder_path, 'metrics.csv'))
    print(f'Save metrics to ./results/{folder_path}/metrics.csv')


def draw(folder):
    # extract predict length from folder name use regex
    path = os.path.join('results', folder)
    pred_len = re.findall(r'pl(\d+)_', folder)[0]
    print(f'Used {path}')

    # load data
    preds = np.load(os.path.join(path, 'pred.npy'))
    trues = np.load(os.path.join(path, 'true.npy'))

    assert preds.shape == trues.shape
    print('Shape', preds.shape)

    n = preds.shape[0]
    indexs = [0, n - 1]
    # extend 10 random indexs, not include 0 and n - 1, and no duplicate
    indexs.extend(np.random.randint(1, n - 1, args.plot_samples))

    for idx in indexs:
        pred = preds[idx, :, :]
        true = trues[idx, :, :]

        plt.figure(figsize=(10, 30))
        for i in range(7):
            plt.subplot(7, 1, i + 1)
            x1 = pred[:, i]
            x2 = true[:, i]
            plt.title(f'Feature {i + 1}')
            plt.plot(x1, label='pred')
            plt.plot(x2, label='true')
            plt.legend()

        saved_path = os.path.join('plots', folder)
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        plt.savefig(f'./{saved_path}/Index_{idx}.png')
        plt.show()
        print(f'Save to ./{saved_path}/Index_{idx}.png')


def plot():
    folders = os.listdir('results')
    folders = sorted(filter(lambda x: x.startswith('lstm') or x.startswith('transformer'), folders))
    length = len(folders)
    while True:
        try:
            for i in range(length):
                print(f'#{i}: {folders[i]}')
            index = input('Select the index of the folder you want to plot: ')
            folder = folders[int(index)]
            draw(folder)
            break
        except Exception as e:
            print(e)
            break


if __name__ == '__main__':
    args = parse_args()

    if args.plot:
        plot()
    else:
        main(args)
