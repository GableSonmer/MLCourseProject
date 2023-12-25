"""
The experiment class.
@author: mysong
@date: 2023/12/20 22:21
@file : exp.py
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LSTMModel, TransformerModel
from oil_dataset import OilDataset
from tools import EarlyStopping, adjust_learning_rate


class Exp:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model_dict = {
            'lstm': LSTMModel,
            'transformer': TransformerModel,
            # add your model here
        }
        model = model_dict[self.args.model](self.args.seq_len,
                                            self.args.label_len,
                                            self.args.pred_len,
                                            self.device)
        return model

    def _get_data(self, flag):
        assert flag in ['train', 'val', 'test']
        batch_size = self.args.batch_size
        if flag == 'test':
            shuffle = False
            drop_last = True
        else:
            shuffle = True
            drop_last = True
        data_set = OilDataset(flag=flag,
                              seq_len=self.args.seq_len,
                              pred_len=self.args.pred_len,
                              label_len=self.args.label_len)
        print(f'Loaded {flag} data, total {len(data_set)}')
        data_loader = DataLoader(data_set,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 drop_last=drop_last,
                                 num_workers=self.args.num_workers)
        return data_set, data_loader

    def _get_optimizer(self):
        optimizer_dict = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
            # add your optimizer here, you should add the optimizer to the parser in main.py
        }
        optimizer = optimizer_dict[self.args.optimizer](self.model.parameters(),
                                                        lr=self.args.lr)
        return optimizer

    def _get_criteria(self):
        criteria_dict = {
            'mse': torch.nn.MSELoss(),
            'mae': torch.nn.L1Loss(),
            # add your criteria here, you should add the criteria to the parser in main.py
        }
        criterion = criteria_dict[self.args.criterion]
        return criterion

    def train(self, setting):
        # set checkpoint path
        path = os.path.join('./checkpoints', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # load data
        train_set, train_loader = self._get_data('train')
        val_set, val_loader = self._get_data('val')
        test_set, test_loader = self._get_data('test')

        # set optimizer and criterion
        optimizer = self._get_optimizer()
        criterion = self._get_criteria()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.epochs):
            train_loss = []
            self.model.train()

            for batch_x, batch_y in tqdm(train_loader):
                # forward
                optimizer.zero_grad()

                # call the _process_one_batch function
                loss_item, _, _ = self._process_one_batch(train_set, batch_x, batch_y, criterion)
                train_loss.append(loss_item)

                optimizer.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(val_set, val_loader, criterion)
            test_loss = self.vali(test_set, test_loader, criterion)
            tqdm.write(f'Epoch {epoch} train loss: {train_loss}, val loss: {vali_loss}, test loss: {test_loss}')

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                tqdm.write("Early stopping")
                break

            # adjust_learning_rate(optimizer, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _process_one_batch(self, dataset_object, batch_x, batch_y, criterion=None):
        """
        Process one batch of data, including forward and backward.
        return the loss.item(), pred, true.
        Because transformer can handle the whole sequence at one time, but lstm can't,
        it needs to be processed one time step and backward one time step.
        Process logic:
            pred, true = self._process_one_batch(train_set, batch_x, batch_y)
            loss = criterion(pred, true)
            # record loss
            train_loss.append(loss.item())
            # backprop
            loss.backward()
        Optional:
            # when use label_len, use this
            # batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
        Return:
            loss_item: the loss.item(), type: float
            pred: the prediction, type: numpy array
            true: the ground truth, type: numpy array
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        # forward
        if self.args.model == 'transformer':
            outputs = self.model(batch_x, batch_y)
        elif self.args.model == 'lstm':
            outputs = self.model(batch_x)
        else:
            raise NotImplementedError

        # backward
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        if criterion is not None:
            loss = criterion(outputs, batch_y)
            loss_item = loss.item()
            torch.backends.cudnn.enabled = False    # 关闭lstm eval时反向传播报错
            loss.backward()
        else:
            loss_item = 0

        # return
        return loss_item, outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy()

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        preds = []
        trues = []
        for batch_x, batch_y in tqdm(test_loader):
            # batch_x: (batch_size, seq_len, feature_dim)
            # batch_y: (batch_size, seq_len+label_len, feature_dim)
            _, pred, true = self._process_one_batch(test_data, batch_x, batch_y)
            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mse = np.mean((preds - trues) ** 2)
        mae = np.mean(np.abs(preds - trues))
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for batch_x, batch_y in tqdm(vali_loader):
            loss, pred, true = self._process_one_batch(vali_data, batch_x, batch_y, criterion)
            # loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
