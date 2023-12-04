import datetime
import numpy as np
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import config
from model import LSTMModel, TransformerModel
from oildataset import OilDataset
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='LSTM')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--O', type=int, default=96, help='predict observation length, 96 or 336')
    parser.add_argument('--model', type=str, default='lstm', help='model name', choices=['lstm', 'transformer'])
    return parser.parse_args()


def load_data():
    train_loader = DataLoader(OilDataset(type='train'), batch_size=64, shuffle=True)
    val_loader = DataLoader(OilDataset(type='val'), batch_size=64, shuffle=False)
    test_loader = DataLoader(OilDataset(type='test'), batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader


def train(args, model, train_loader, val_loader):
    # 参数设置
    sequence_length = args.O
    learning_rate = args.lr
    num_epochs = args.epochs

    # 实例化模型、损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    max_loss = float('inf')

    # 绘制训练过程中的loss曲线
    loss_list = []

    for epoch in range(num_epochs):
        # train
        running_loss = 0.0
        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.view(-1, sequence_length, config.input_size)
            labels = labels.view(-1, 1)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 记录loss
            running_loss += loss.item()

        # 添加loss到loss_list
        loss_list.append(running_loss / len(train_loader))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}')

        # val
        model.eval()
        with torch.no_grad():
            losses = []
            for sequences, labels in val_loader:
                sequences = sequences.view(-1, sequence_length, config.input_size)
                labels = labels.view(-1, 1)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
            print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {np.mean(losses):.4f}')
            val_loss = np.mean(losses)
            if val_loss < max_loss:
                print(f'{max_loss} -> {val_loss} Saving model...')
                max_loss = val_loss
                torch.save(model.state_dict(), 'lstm.pth')

    # 绘制loss曲线
    plt.plot(loss_list)
    now = datetime.datetime.now()
    y_m_d = now.strftime('%Y-%m-%d')
    plt.savefig(f'./outputs/{args.model}_{y_m_d}_{args.epochs}_loss.png')
    plt.show()


def test(args, model, test_loader):
    sequence_length = args.O
    # test
    model.eval()
    criterion1 = nn.L1Loss()
    criterion2 = nn.MSELoss()
    with torch.no_grad():
        losses1 = []
        losses2 = []
        for sequences, labels in test_loader:
            sequences = sequences.view(-1, sequence_length, config.input_size)
            labels = labels.view(-1, 1)
            outputs = model(sequences)
            loss1 = criterion1(outputs, labels)
            loss2 = criterion2(outputs, labels)
            losses1.append(loss1.item())
            losses2.append(loss2.item())
        print(f'Test MAE: {np.mean(losses1):.4f}, Test MSE: {np.mean(losses2):.4f}')


if __name__ == '__main__':
    args = parse_args()
    train_loader, val_loader, test_loader = load_data()
    if args.model == 'lstm':
        model = LSTMModel(config.input_size)
    else:
        model = TransformerModel(config.input_size)
    train(args, model, train_loader, val_loader)
    test(args, model, test_loader)
