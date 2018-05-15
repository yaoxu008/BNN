from __future__ import division
import os
import time
from six.moves import xrange
from ops import *
import load
import util
import cca
import numpy as np


# --------------------------------
#             Main
# --------------------------------

class BNN(object):
    def __init__(self, sess,
                 batch_size=8, conv_dim='20,40,50', conv_kernel='3,3,3', out_dim=50,
                 dataset_name='mnist', classify='False',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, alpha=1.0, NP_ratio=9, train_size=1000,
                 test_size=200):

        self.sess = sess

        self.batch_size = batch_size
        self.classify = classify
        self.conv_dim = conv_dim.split(',')
        self.conv_kernel = conv_kernel.split(',')
        self.out_dim = out_dim
        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.alpha = alpha
        self.NP_ratio = NP_ratio
        self.train_size = train_size
        self.test_size = test_size

        if self.dataset_name == 'mnist':
            self.input_height_l = 28
            self.input_width_l = 14
            self.input_height_r = 28
            self.input_width_r = 14
            self.trSet, self.teSet = load.mnist(self.train_size, self.test_size)
            self.trlabel, self.telabel = load.mnist_label(self.train_size, self.test_size)

            self.data_X, self.data_Y, self.data_Flag = load.pairs_generate(self.trSet, self.NP_ratio,
                                                                           self.trlabel, classify)
            self.Test_X, self.Test_Y, self.Test_Flag = load.pairs_generate(self.teSet, 2,
                                                                           self.telabel, classify)
            self.c_dim = 1

        elif dataset_name == 'xrmb':
            self.input_height_l = 1
            self.input_width_l = 273
            self.input_height_r = 1
            self.input_width_r = 112

            self.X1, self.X2 = load.loadmat(self.train_size + self.test_size)
            self.data_X, self.data_Y, self.data_Flag = load.pair_xrmb(self.X1[:train_size], self.X2[:train_size],
                                                                      self.NP_ratio)

            self.Test_X, self.Test_Y, self.Test_Flag = load.pair_xrmb(self.X1[train_size:], self.X2[train_size:],
                                                                      2)

            self.c_dim = 1

        self.build_model()

    def build_model(self):

        image_dims_l = [self.input_height_l, self.input_width_l, self.c_dim]
        image_dims_r = [self.input_height_r, self.input_width_r, self.c_dim]

        self.l_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims_l, name='left_data')

        self.r_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims_r, name='right_data')

        self.flags = tf.placeholder(
            tf.float32, [self.batch_size, 1], name='flag')

        l_inputs = self.l_inputs
        r_inputs = self.r_inputs
        flags = self.flags

        self.L = self.left_cnn(l_inputs, reuse=False)
        self.R = self.right_cnn(r_inputs, reuse=False)

        self.l_sum = histogram_summary("l", self.L)
        self.r_sum = histogram_summary("r", self.R)

        self.infer_result = Euclidean(self.L, self.R)

        self.l_loss = Lossfunction2(self.L, self.R, self.flags, self.alpha, self.NP_ratio, self.batch_size)

        self.r_loss = Lossfunction2(self.L, self.R, self.flags, self.alpha, self.NP_ratio, self.batch_size)

        self.l_loss_sum = scalar_summary("l_loss", self.l_loss)
        self.r_loss_sum = scalar_summary("r_loss", self.r_loss)

        t_vars = tf.trainable_variables()

        self.l_vars = [var for var in t_vars if 'l_' in var.name]
        self.r_vars = [var for var in t_vars if 'r_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        global_step = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(config.learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=10,
                                                   decay_rate=config.decay)

        l_optim = tf.train.GradientDescentOptimizer(learning_rate) \
            .minimize(self.l_loss, var_list=self.l_vars)
        r_optim = tf.train.GradientDescentOptimizer(learning_rate) \
            .minimize(self.r_loss, var_list=self.r_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.l_sum = merge_summary([self.l_sum, self.l_loss_sum])
        self.r_sum = merge_summary([self.r_sum, self.r_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        # loading from checkpoint
        could_load, checkpoint_epoch = self.load(self.checkpoint_dir)
        if could_load:
            epoch = checkpoint_epoch
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        current_lr = config.learning_rate
        errL = errR = 1

        # Creat log file
        f_result = open(self.log_dir(), 'w')

        # start training
        for epoch in xrange(config.epoch):
            batch_idxs = len(self.data_X) // config.batch_size

            # shuffle the dataset
            util.shuffle(self.data_X, self.data_Y, self.data_Flag)
            write = False

            if np.mod(epoch, config.feedback_epoch) == 0:
                print("=======================Test Info on conv_size =  " + str(
                    self.conv_dim) + "=========================")
                print("### Testing at epoch %2d... " % epoch)
                print("### Learning rate: %4f..." % current_lr)
                currmax = avg_corr = 0

                if self.classify == False:
                    currmax, avg_corr = self.doCCA(config.batch_size, self.out_dim)

                accuracy_avg1 = self.inference(False, config, 0, 0.7, epoch)
                accuracy_avg2 = self.inference(False, config, 0, 0.5, epoch)
                accuracy_avg3 = self.inference(False, config, 0, 0.3, epoch)

            for idx in xrange(batch_idxs):
                batch_X = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_Y = self.data_Y[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_Flag = self.data_Flag[idx * config.batch_size:(idx + 1) * config.batch_size]

                # Update Left network
                for inner in xrange(0, config.inner):
                    _, summary_str = self.sess.run([l_optim, self.l_sum],
                                                   feed_dict={
                                                       self.l_inputs: batch_X,
                                                       self.r_inputs: batch_Y,
                                                       self.flags: batch_Flag
                                                   })
                    self.writer.add_summary(summary_str, epoch)

                errL = self.r_loss.eval({
                    self.l_inputs: batch_X,
                    self.r_inputs: batch_Y,
                    self.flags: batch_Flag
                })

                # Update Right network
                for inner in xrange(0, config.inner):
                    _, summary_str = self.sess.run([r_optim, self.r_sum],
                                                   feed_dict={
                                                       self.l_inputs: batch_X,
                                                       self.r_inputs: batch_Y,
                                                       self.flags: batch_Flag
                                                   })
                    self.writer.add_summary(summary_str, epoch)

                errR = self.l_loss.eval({
                    self.l_inputs: batch_X,
                    self.r_inputs: batch_Y,
                    self.flags: batch_Flag
                })

                if np.mod(counter, 1000) == 1:
                    print("Epoch:[%2d/%2d]==Batch_id:[%2d/%2d]==time:%4.4f==loss: %.4f (left) and %.4f (right)" \
                          % (epoch, config.epoch, idx, batch_idxs,
                             time.time() - start_time, errL, errR))

                counter += 1

            # update learning rate
            add_global = global_step.assign_add(1)
            _, current_lr = self.sess.run([add_global, learning_rate])

            if np.mod(epoch, config.feedback_epoch) == 0:
                input_data = str(epoch) + "\t" + str(errL) + "\t" + str(avg_corr) + "\t" + str(currmax) + "\t" + str(
                    accuracy_avg1) + "\t" + str(accuracy_avg2) + "\t" + str(accuracy_avg3) + "\n"
                f_result.write(input_data)
                f_result.flush()

            if self.classify == True:
                if np.mod(epoch, 100) == 0:
                    self.savePosCR(config.batch_size, epoch)

            if np.mod(epoch, 20) == 0 and epoch != 0:
                # reselect neg_data
                if self.dataset_name == 'mnist':
                    self.data_X, self.data_Y, self.data_Flag = load.pairs_2(self.trSet, self.NP_ratio, self.trlabel)
                elif self.dataset_name == 'xrmb':
                    self.data_X, self.data_Y, self.data_Flag = load.pair_xrmb(self.X1[:self.train_size],
                                                                              self.X2[:self.train_size], self.NP_ratio)
                self.save(config.checkpoint_dir, epoch)

        f_result.close()

    def savePosCR(self, batch_size, epoch, write=True):
        telefts, terights, teCorr = load.mnist_Pospairs(self.teSet, shuffle=False)
        batch_idxs = len(telefts) // batch_size
        if write:
            feature_dir = "features/"
            if not os.path.exists(feature_dir):
                os.makedirs(feature_dir)

            L_file = open(feature_dir + str(epoch) + "_L", 'w')
            R_file = open(feature_dir + str(epoch) + "_R", 'w')

        for idx in xrange(batch_idxs):

            Lval, Rval = self.sess.run([self.L, self.R], feed_dict={
                self.l_inputs: telefts[idx * batch_size:(idx + 1) * batch_size],
                self.r_inputs: terights[idx * batch_size:(idx + 1) * batch_size],
                self.flags: teCorr[idx * batch_size:(idx + 1) * batch_size]
            })

            if write:
                for li in Lval:
                    for lii in li:
                        L_file.write(str(lii) + ' ')
                    L_file.write("\n")
                for ri in Rval:
                    for rii in ri:
                        R_file.write(str(rii) + ' ')
                    R_file.write("\n")

        if write:
            L_file.close()
            R_file.close()

    def doCCA(self, batch_size, corr_size):
        if self.dataset_name == 'mnist':
            telefts, terights, teCorr = load.mnist_Pospairs(self.teSet, shuffle=True)
        elif self.dataset_name == 'xrmb':
            telefts = self.X1[self.train_size:]
            terights = self.X2[self.train_size:]
            teCorr = np.ones([telefts.shape[0], 1])
            telefts, terights, teCorr = util.shuffle(telefts, terights, teCorr)
        batch_idxs = len(telefts) // batch_size
        currmax = 0
        sum = 0
        counter = 0

        for idx in xrange(batch_idxs):

            Lval, Rval = self.sess.run([self.L, self.R], feed_dict={
                self.l_inputs: telefts[idx * batch_size:(idx + 1) * batch_size],
                self.r_inputs: terights[idx * batch_size:(idx + 1) * batch_size],
                self.flags: teCorr[idx * batch_size:(idx + 1) * batch_size]
            })

            corr_idxs = batch_size // corr_size
            for corrx in xrange(corr_idxs):
                Lval_i = Lval[corrx * corr_size:(corrx + 1) * corr_size]
                Rval_i = Rval[corrx * corr_size:(corrx + 1) * corr_size]
                try:
                    A, B, Corr = cca.cca(Lval_i, Rval_i, self.out_dim)
                except:
                    counter -= 1
                    Corr = 0
                    pass
                currmax = max(currmax, Corr)
                sum = sum + Corr
                counter += 1

        avg_corr = sum / counter
        # print("### Average Correlation is: %.4f" % avg_corr)
        print("### Maximum Correlation is: %.4f" % currmax)
        return currmax, avg_corr

    def inference(self, ifload, config, tridx, threshold, epoch):
        accuracy_avg = 0
        positive_avg = 0
        negative_avg = 0

        if ifload:
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            if could_load:
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        test_result = np.array([])

        batch_idxs = len(self.Test_X) // config.batch_size

        for idx in xrange(0, batch_idxs):
            test_X = self.Test_X[idx * config.batch_size:(idx + 1) * config.batch_size]
            test_Y = self.Test_Y[idx * config.batch_size:(idx + 1) * config.batch_size]
            test_Flag = self.Test_Flag[idx * config.batch_size:(idx + 1) * config.batch_size]
            result = self.sess.run(self.infer_result, feed_dict={
                self.l_inputs: test_X,
                self.r_inputs: test_Y,
                self.flags: test_Flag
            })
            result = util.YesNo(result, threshold).reshape([-1, 1])
            test_result = np.append(test_result, result)
            positive_avg = positive_avg + np.mean(result[test_Flag == 1])
            negative_avg = negative_avg + np.mean(result[test_Flag == 0])
            difference = np.abs(result - (test_Flag * -1 + 1))
            accuracy = np.sum(difference) / test_Flag.shape[0]
            accuracy_avg = accuracy_avg + accuracy

        # accuracy_avg = accuracy_avg / batch_idxs
        positive_avg = positive_avg / batch_idxs
        negative_avg = negative_avg / batch_idxs
        accuracy_avg = ((1 - positive_avg) + self.alpha * negative_avg) / (1 + self.alpha)
        '''
        print(
            "### Accuracy on Testing Sets: %.4f(total) ###  %.2f(threshold)###  %.4f(postive) ### %.4f(negative) ### NP_ratio:%.2f"
            % (accuracy_avg, threshold, (1 - positive_avg), negative_avg, self.NP_ratio))
        '''
        if threshold == 0.5:
            test_result = test_result.reshape([-1, 1])
            test_size = batch_idxs * config.batch_size
            Test_Flag = self.Test_Flag[:test_size]
            Test_X = self.Test_X[:test_size]
            Test_Y = self.Test_Y[:test_size]
            real_true_label = np.logical_and(Test_Flag == 1, test_result == 0).reshape([-1])
            fake_true_label = np.logical_and(Test_Flag == 0, test_result == 0).reshape([-1])
            real_false_label = np.logical_and(Test_Flag == 0, test_result == 1).reshape([-1])
            fake_false_label = np.logical_and(Test_Flag == 1, test_result == 1).reshape([-1])

            TP = real_true_label[real_true_label == True].shape[0]
            TN = real_false_label[real_false_label == True].shape[0]
            FP = fake_true_label[fake_true_label == True].shape[0]
            FN = fake_false_label[fake_false_label == True].shape[0]

            print('TP = %2d, TN = %2d, FP = %2d, FN= %2d' % (TP, TN, FP, FN))
            if (TP != 0 and FP != 0):
                AC = (TP + TN) / (TP + TN + FP + FN)
                RC = (TP) / (TP + FN)
                PC = TP / (TP + FP)
                print('accuracy = %.4f, precision = %.4f, recall = %.4f' % (AC, PC, RC))

            if self.dataset_name == 'mnist' and np.mod(epoch, 40) == 0:
                real_true_l = Test_X[real_true_label]
                real_true_r = Test_Y[real_true_label]
                fake_true_l = Test_X[fake_true_label]
                fake_true_r = Test_Y[fake_true_label]
                real_false_l = Test_X[real_false_label]
                real_false_r = Test_Y[real_false_label]
                fake_false_l = Test_X[fake_false_label]
                fake_false_r = Test_Y[fake_false_label]

                util.savimg(real_true_l, real_true_r, 10, 'real_true_' + str(epoch))
                util.savimg(fake_true_l, fake_true_r, 10, 'fake_true_' + str(epoch))
                util.savimg(real_false_l, real_false_r, 10, 'real_false_' + str(epoch))
                util.savimg(fake_false_l, fake_false_r, 10, 'fake_false_' + str(epoch))

        return accuracy_avg

    def left_cnn(self, image, reuse=False):
        with tf.variable_scope("leftcnn") as scope:
            if reuse:
                scope.reuse_variables()
            if self.dataset_name == 'mnist':
                h0 = mul_conv2d(image, 'l_', self.conv_dim, self.conv_kernel, self.conv_kernel,
                                d_h=1, d_w=1, stddev=0.02, name="l_mul_conv2d")
            elif self.dataset_name == 'xrmb':
                kernal_one = np.ones(len(self.conv_kernel))
                h0 = mul_conv2d(image, 'l_', self.conv_dim, kernal_one, self.conv_kernel,
                                d_h=1, d_w=1, stddev=0.02, name="l_mul_conv2d")

            h1 = linear(tf.reshape(h0, [self.batch_size, -1]), self.out_dim, 'l_h2_lin')
            return tf.nn.sigmoid(h1)

    def right_cnn(self, image, reuse=False):
        with tf.variable_scope("rightcnn") as scope:
            if reuse:
                scope.reuse_variables()

            if self.dataset_name == 'mnist':
                h0 = mul_conv2d(image, 'l_', self.conv_dim, self.conv_kernel, self.conv_kernel,
                                d_h=1, d_w=1, stddev=0.02, name="l_mul_conv2d")
            elif self.dataset_name == 'xrmb':
                kernal_one = np.ones(len(self.conv_kernel))
                h0 = mul_conv2d(image, 'l_', self.conv_dim, kernal_one, self.conv_kernel,
                                d_h=1, d_w=1, stddev=0.02, name="l_mul_conv2d")
            h1 = linear(tf.reshape(h0, [self.batch_size, -1]), self.out_dim, 'r_h2_lin')
            return tf.nn.sigmoid(h1)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            str(self.conv_dim).replace('[', '').replace(']', ''),
            str(self.conv_kernel).replace('[', '').replace(']', ''),
            self.out_dim)

    def save(self, checkpoint_dir, step):
        model_name = "BNN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def log_dir(self):
        log_dir_ = "{}_{}_{}_{}_{}_{}.log".format(
            self.dataset_name, self.batch_size, self.conv_dim, self.conv_kernel, self.out_dim, self.NP_ratio)
        log_dir_ = os.path.join("logs", log_dir_)
        # if not os.path.exists(log_dir_):
        # os.maknod(checkpoint_dir)
        return str(os.path.join(log_dir_))

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            epoch = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, epoch
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
