import os
import sys
import argparse

import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import logging

import argparse

import time
from datetime import datetime

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import Dataset, DataLoader

import model.BNN as BNN

from model.data import DATA_PAIR
import random
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.01, type=float, help='learning rate decay')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=32, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=16, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--out_dim', default=20, type=int, help='feature dimension')
    parser.add_argument('--alpha', default=1, type=float, help='balance of neg vs pos')
    parser.add_argument('--NP_ratio', default=1, type=float, help='ratio of neg/pos in training set')
    parser.add_argument('--input_channel1', default=3, type=int, help='chanel1_indim')
    parser.add_argument('--input_channel2', default=3, type=int, help='chanel2_indim')

    args = parser.parse_args()
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True

    solver = Solver(args)
    solver.train()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.decay = config.decay
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.out_dim = config.out_dim
        self.alpha = config.alpha
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.np_ratio = config.NP_ratio
        self.resnet = config.resnet
        self.input_channel1 = config.input_channel1
        self.input_channel2 = config.input_channel2
        if self.resnet == 50:
            self.resnet50 = True
        elif self.resnet == 101:
            self.resnet50 = False
        else:
            print('only support resnet50 or 101')
            raise
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
            torch.cuda.set_device(1)
        else:
            self.device = torch.device('cpu')

        self.model = BNN.BNN(self.out_dim, self.alpha, self.input_channel1, self.input_channel2).to(self.device)
        self.optimizer_1 = optim.SGD(self.model.parameters_1, lr=self.lr)
        self.optimizer_2 = optim.SGD(self.model.parameters_2, lr=self.lr)
        self.optimizer_total = optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer_total, step_size=5, gamma=self.decay)

        self.train_data = DATA_PAIR("data/train")
        self.test_data = DATA_PAIR("data/test")

        self.test_data.reset(1)
        self.testloader = DataLoader(self.test_data, batch_size=self.test_batch_size, shuffle=True)

        # 日志存储
        filepath = 'log'
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.logfilename = os.path.join(filepath, 'log' + str(self.out_dim) + '_' + str(self.lr) + '_' + str(
            self.train_batch_size) + '_' + str(self.resnet) + 'eval.txt')
        self.testfilename = os.path.join(filepath, 'log' + str(self.out_dim) + '_' + str(self.lr) + '_' + str(
            self.train_batch_size) + '_' + str(self.resnet) + 'eval_test.txt')
        self.logfile = open(self.logfilename, "w")
        self.logfile.writelines('epoch, batch, loss\n')
        self.testfile = open(self.testfilename, "w")

    def train(self):

        save_checkpoint = 'models/checkpoint_' + str(self.out_dim) + '_' + 'eval.pth'

        backbone_checkpoint = 'models/backbone_' + str(self.out_dim) + '_' + str(self.lr) + '_' + str(
            self.train_batch_size) + '_' + str(self.resnet)+'.pth'

        # From imagenet
        if os.path.exists(backbone_checkpoint):
            print('recovering parameters from pretrain model')
            self.model.model_1.backbone.load_state_dict(torch.load(backbone_checkpoint), strict=False)
            self.model.model_2.backbone.load_state_dict(torch.load(backbone_checkpoint), strict=False)

        # continue training
        if os.path.exists(save_checkpoint):
            print('continue training')
            self.model.load_state_dict(torch.load(save_checkpoint), strict=True)

        for epoch in range(self.epochs):
            print('preparing data')
            self.train_data.reset(self.np_ratio)
            trainloader = DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True)
            print('preparing data complete')
            if epoch % 10 == 0:
                self.test()
            self.model.train()

            for i, (batch_x1, batch_x2, batch_y) in enumerate(trainloader, 0):

                # epoch_start_time = time.time()
                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()
                self.optimizer_total.zero_grad()

                batch_x1 = batch_x1.float().to(self.device)
                batch_x2 = batch_x2.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                D4, p_d, n_d, distance = self.model(batch_x1, batch_x2, batch_y)
                Loss = D4

                # For seperate training
                Loss.backward(retain_graph=True)
                self.optimizer_1.step()
                Loss.backward()
                self.optimizer_2.step()


                self.logfile.writelines('%d, %d, %.2f\n' % (epoch, i, Loss))
                self.logfile.flush()
                if torch.isnan(Loss):
                    print('Loss = NaN!!!')
                    sys.exit(0)

                if np.mod(i, 10) == 0 and i != 0:
                    print('epoch:{:d} batch:{:d} loss_bnn: {:.4f} loss_p: {:.4f} loss_n: {:.4f}'.format(epoch, i, Loss,
                                                                                                        p_d, n_d))

            self.scheduler.step(epoch)
            print('saving model')
            torch.save(self.model.state_dict(), save_checkpoint)
            torch.save(self.model.model_1.state_dict(), backbone_checkpoint)

    def test(self):
        print('testing')

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        self.model.eval()

        for i, (test_batch_x1, test_batch_x2, test_batch_y) in enumerate(self.testloader, 0):
            test_batch_x1 = test_batch_x1.float().to(self.device)
            test_batch_x2 = test_batch_x2.float().to(self.device)
            test_batch_y = test_batch_y.float().to(self.device)

            D4, p_d, n_d, distance = self.model(test_batch_x1, test_batch_x2, test_batch_y)

            predict = distance > 0.38


            predict = torch.tensor([float(i) for i in predict]).to(self.device)

            predict_ = predict * -1 + 1

            TP += predict_[test_batch_y == 1].sum()
            FP += predict_[test_batch_y == 0].sum()
            TN += predict[test_batch_y == 0].sum()
            FN += predict[test_batch_y == 1].sum()

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP)
        neg_precision = TN / (TN + FN)
        recall = TP / (TP + FN)

        self.testfile.writelines('accuracy:%.4f, precision:%.4f, neg_precision:%.4f, recall:%.4f\n' % (
            accuracy, precision, neg_precision, recall))
        self.testfile.flush()
        print('accuracy: {:.4f} precision: {:.4f} neg_precision: {:.4f} recall: {:.4f}'.format(accuracy, precision,
                                                                                               neg_precision, recall))
        print('TP: {:.1f} TN: {:.1f} FP: {:.1f} FN: {:.1f}'.format(TP, TN, FP, FN))
        return 0


if __name__ == '__main__':
    main()
