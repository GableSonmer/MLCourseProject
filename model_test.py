import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from model import LSTMModel, TransformerModel
from arguments import args
import joblib


def result_plot(args):
    dev = args.dev
    model = LSTMModel(7, args).to(args.dev)
    model.load_state_dict(torch.load('./saved_models/lstm_100.pth'))

    data = pd.read_csv('data/ETTh1.csv', sep=',').head(96 * 2).drop(columns=['date'])
    predict_data = torch.tensor(data[:96].values, dtype=torch.float32)
    labels = data.values

    # load scaler for inverse transform
    scaler = joblib.load('scaler.pkl')

    model.eval()
    with torch.no_grad():
        sequences = predict_data.view(1, 96, 7)

        sequences = sequences.to(dev)

        outputs = model(sequences)
        predict_date = range(96, 96 + args.O)
        true_date = range(96 + args.O)
        for i in range(7):
            plt.subplot(7, 1, i + 1)
            plt.plot(predict_date, outputs[i, :, :].cpu().numpy().T, label=f'Prediction{i}')
            plt.plot(true_date, labels[:, i].T, label=f'Ground Truth{i}')
            plt.legend()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    result_plot(args)
