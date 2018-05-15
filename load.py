import scipy.io as sio
import numpy as np
import os
import util

# ------------------------------------------
#  Load and generate MNIST or XRMB samples
# ------------------------------------------

mnist_dir = 'data/mnist/'
xrmb_dir = 'data/xrmb/'


# load xrmb data
def loadmat(num):
    print('Fetching XRMB data')
    file1 = xrmb_dir + 'XRMBf2KALDI_window7_single1.mat'
    file2 = xrmb_dir + 'XRMBf2KALDI_window7_single2.mat'
    data1 = sio.loadmat(file1)
    data2 = sio.loadmat(file2)
    dataX = data1['X1']
    dataY = data2['X2']
    index = np.arange(dataX.shape[0])
    print('Shuffle the data')
    np.random.shuffle(index)
    dataX = dataX[index[:num]]
    dataY = dataY[index[:num]]

    X1 = np.array(dataX).reshape(num, 1, 273, 1)
    X2 = np.array(dataY).reshape(num, 1, 112, 1)

    return X1, X2


# load mnist dataset
def mnist(num1, num2):
    print("Fetching data with size %2d (train) and %2d (test)" % (num1, num2))

    fd = open(os.path.join(mnist_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trSet = loaded[16:(16 + 28 * 28 * num1)].reshape((num1, 28 * 28)).astype(float)

    fd = open(os.path.join(mnist_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teSet = loaded[16:(16 + 28 * 28 * num2)].reshape((num2, 28 * 28)).astype(float)

    return trSet, teSet


'''
# generate mnist pairs
def mnist_pairs(neg_ratio, num1, num2):
    trSet, teSet = mnist(num1, num2)

    print("Generating data with negtive ratio %2d" % neg_ratio)
    # trSet = util.shuffle(trSet)
    trlefts, trrights, trnum = util.imgdiv(trSet, 28, 28, 1)
    trX, trY, trCorr = util.pairing(trlefts, trrights, trnum, neg_ratio)
    telefts, terights, tenum = util.imgdiv(teSet, 28, 28, 1)
    teX, teY, teCorr = util.pairing(telefts, terights, tenum, neg_ratio)

    return trX, trY, trCorr, teX, teY, teCorr
'''


def pairs_generate(data, neg_ratio, label, classify=False):
    print("Generating data with negtive ratio %2d" % neg_ratio)
    # trSet = util.shuffle(trSet)
    X_, Y_, num = util.imgdiv(data, 28, 28, 1)
    X, Y, Corr1, Corr3 = util.pairing_2(X_, Y_, num, neg_ratio, label)

    if classify:
        return X, Y, Corr3
    else:
        return X, Y, Corr1


def pair_xrmb(lefts, rights, neg_ratio):
    print("Generating negative samples...")
    num = lefts.shape[0]
    index_x = np.arange(num)
    index_x_ = index_x.repeat(1 + neg_ratio, axis=0)
    lefts_ = lefts[index_x_]

    index_y = np.random.randint(0, num, [num, neg_ratio])
    index_y_ = np.append(np.arange(num).reshape(num, 1), index_y, 1)
    rights_ = rights[index_y_.reshape(index_x_.shape)]
    corr1 = index_y_ == index_y_[:, 0].reshape(num, 1)
    corr1 = np.array(corr1).astype(float).reshape(num * (1 + neg_ratio), 1)

    return lefts_, rights_, corr1


def mnist_Pospairs(teSet, shuffle=False):
    if shuffle:
        np.random.shuffle(teSet)
    telefts, terights, tenum = util.imgdiv(teSet, 28, 28, 1)
    teCorr = np.ones([tenum, 1])
    return telefts, terights, teCorr


def flagGen(flags_, alpha):
    # group_num = batch_size/(1 + NP_ratio)
    if alpha > 1:
        flags = flags_ / alpha + (flags_ * (-1) + 1)
    else:
        flags = flags_ + (flags_ * (-1) + 1) * alpha
    return flags


def mnist_label(num1, num2):
    print("Fetching mnist_label with size %2d (train) and %2d (test)" % (num1, num2))

    fd = open(os.path.join(mnist_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(mnist_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trLabel = np.asarray(trY)
    teLabel = np.asarray(teY)

    return trLabel[:num1], teLabel[:num2]
