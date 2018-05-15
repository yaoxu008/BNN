# --------------------------------------------
#  Independent Function for Transfer Learning
# --------------------------------------------

import time
from sklearn import metrics
import numpy as np
import load

logits_dir = "data/logits/"


def loadData(dir):
    data_X = np.loadtxt(dir + "L")[:10000]
    data_Y = np.loadtxt(dir + "R")[:10000]
    data_label, _ = load.mnist_label(10000, 0)
    print(data_label.shape)

    return data_X, data_Y, data_label


def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='linear', probability=False, C=1)
    model.fit(train_x, train_y)
    return model


def fivefold(data_X, data_Y, data_label):
    avg_accuracy = 0
    num_train, num_feat = data_X.shape
    num_test, _ = data_Y.shape
    fold_num = num_train // 5
    print('#data X: %d, #data Y: %d, dimension: %d, fold number: %d' % (num_train, num_test, num_feat, fold_num))
    for i in range(5):
        train_x = np.append(data_X[:i * fold_num], data_X[(i + 1) * fold_num:], 0)
        train_y = np.append(data_label[:i * fold_num], data_label[(i + 1) * fold_num:], 0)
        test_x = data_Y[i * fold_num:(i + 1) * fold_num]
        test_y = data_label[i * fold_num:(i + 1) * fold_num]

        start_time = time.time()
        model = svm_classifier(train_x, train_y)

        print('Fold %d training took %fs!' % (i + 1, time.time() - start_time))

        predict = model.predict(test_x)

        accuracy = metrics.accuracy_score(test_y, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))

        avg_accuracy = avg_accuracy + accuracy
    avg_accuracy = avg_accuracy / 5
    print('# Average accuracy of 5-fold test case: %.2f%%' % (100 * avg_accuracy))

    return avg_accuracy


print('reading training and testing data...')
data_X, data_Y, data_label = loadData(logits_dir)

print('\n********* Transfer learning: Left to Right *********')
fivefold(data_X, data_Y, data_label)
print('\n********* Transfer learning: Right to Left *********')
fivefold(data_Y, data_X, data_label)

print('\n********* Transfer learning: Left Single View *********')
fivefold(data_X, data_X, data_label)
print('\n********* Transfer learning: Right Single View *********')
fivefold(data_Y, data_Y, data_label)
