import numpy as np
from scipy import misc


def imgdiv(data, height, width, channel):
    print("///Generating pairs...")
    num, img = np.shape(data)
    imgs = data.reshape(num, height, width, channel)
    lefts = imgs[:, :, :int(width / 2), :]
    rights = imgs[:, :, int(width / 2):, :]
    lefts = lefts.reshape(num, height, int(width / 2), channel)
    rights = rights.reshape(num, height, int(width / 2), channel)
    print("///finish pairing...")

    return lefts, rights, num


def pairing(lefts, rights, pos_num, neg_num):
    print("Generating negative samples...")

    lefts_ = lefts.repeat(1 + neg_num, axis=0)

    r_index = np.array([[]], dtype=int)
    r_index_ = np.arange(pos_num)

    for i in range(pos_num):
        temp = np.array([i])
        temp_ = np.delete(r_index_, i)
        np.random.shuffle(temp_)
        temp_ = temp_[:neg_num]
        temp = np.append(temp, temp_)
        temp = temp.reshape([1, 1 + neg_num])
        r_index = np.append(r_index, temp)

    rights_ = rights[r_index]

    corr = np.ones([pos_num, 1])
    corr_ = np.zeros([pos_num, neg_num])
    corr = np.append(corr, corr_, 1)
    corr = np.reshape(corr, [pos_num * (1 + neg_num), 1])

    return lefts_, rights_, corr


def pairing_2(lefts, rights, num, neg_num, label):
    print("Generating negative samples...")
    lefts_ = lefts.repeat(1 + neg_num, axis=0)

    index_y = np.random.randint(0, num, [num, neg_num])
    index_y_ = np.append(np.arange(num).reshape(num, 1), index_y, 1)
    rights_ = rights[index_y_].reshape(lefts_.shape)
    corr1 = index_y_ == index_y_[:, 0].reshape(num, 1)
    corr1 = np.array(corr1).astype(float).reshape(num * (1 + neg_num), 1)
    label_ = label[index_y_]
    corr3 = label_ == label_[:, 0].reshape(num, 1)
    corr3 = np.array(corr3).astype(float).reshape(num * (1 + neg_num), 1)

    return lefts_, rights_, corr1, corr3


def shuffle(dataX, dataY, label):
    index = np.arange(len(dataX))
    np.random.shuffle(index)
    dataX = dataX[index]
    dataY = dataY[index]
    label = label[index]
    return dataX, dataY, label


def YesNo(data, threshold):
    temp = data > threshold
    return temp * 1


def savimg(leftimg, rightimg, N, name):
    num, height, width, channel = leftimg.shape
    imgs_ = np.append(leftimg, rightimg, 2)

    imgs = np.zeros([height * N, width * 2 * N, channel])
    for i in range(N):
        for j in range(N):
            if i * N + j < imgs_.shape[0]:
                imgs[i * height:(i + 1) * height, j * width * 2:(j + 1) * width * 2] = imgs_[i * N + j]
    if channel == 1:
        imgs = imgs.reshape([height * N, width * 2 * N])

    misc.toimage(imgs).save('samples/' + name + '.jpg')

    return imgs
