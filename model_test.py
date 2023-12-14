################## IMPORTANT ###########################
# Precondition:   Run main.py first to train.          #
# Not-Allowed:    Modify the model or arguments.py.    #
# Post-condition: Plot the prediction and ground truth.#
########################################################

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import arguments
from model import LSTMModel, TransformerModel
from arguments import args
import joblib


def result_plot(args):
    dev = args.dev
    if args.model == 'lstm':
        model = LSTMModel(7).to(args.dev)
    elif args.model == 'transformer':
        model = TransformerModel(7, 7).to(args.dev)
    else:
        raise TypeError('No such model!')

    model_path = f'./saved_models/{args.model}_{args.epochs}.pth'
    model.load_state_dict(torch.load(model_path))

    # load scaler for inverse transform
    scaler = joblib.load('scaler.pkl')

    data = pd.read_csv('data/ETTh1.csv', sep=',').drop(columns=['date'])

    # two tasks: predict_sequence_length = 96 or 336
    # use 96 to predict predict_sequence_length
    predict_sequence_length = 96
    max_start_index = len(data) - 96 - predict_sequence_length + 1
    start_index = np.random.randint(0, max_start_index)

    # print some information
    print(f'Predict sequence length: {predict_sequence_length}')
    print(f'Select range: {start_index} - {start_index + predict_sequence_length * 2}')

    data = data.iloc[start_index:start_index + 96 + predict_sequence_length, :]
    data = scaler.transform(data)

    predict_data = torch.tensor(data[:96], dtype=torch.float32).to(dev)
    labels = scaler.inverse_transform(data)

    # predict
    model.eval()
    with torch.no_grad():
        for i in range(predict_sequence_length):
            sequences = predict_data[i:96 + i].view(1, 96, 7)
            sequences = sequences.to(dev)
            outputs = model(sequences)
            predict_data = torch.cat([predict_data, outputs], dim=0)

    # draw the figure
    plt.figure(figsize=(10, 30))
    predict_data = scaler.inverse_transform(predict_data.cpu())
    x = range(96 + predict_sequence_length)
    for i in range(7):
        plt.subplot(7, 1, i + 1)
        plt.title(f'Feature <{arguments.COLUMN_NAMES[i]}>')
        plt.plot(x, predict_data[:, i].T, label=f'Prediction{i}')
        plt.plot(x, labels[:, i].T, label=f'Ground Truth{i}')
        plt.legend(loc='upper left', prop={'size': 4})

    fig_path = f'./outputs/{args.model}_{args.epochs}_{predict_sequence_length}_predict_ground_truth.png'
    print('Saved to', fig_path)
    plt.savefig(fig_path, dpi=150)
    plt.show()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    result_plot(args)
