import os
from model import BNN
import tensorflow as tf

# --------------------------------
#          configuration
# --------------------------------
flags = tf.app.flags
flags.DEFINE_integer("epoch", 2000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.02, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("decay", 0.99, "decay of learning rate [0.98]")
flags.DEFINE_integer("train_size", 60000, "The size of train image pairs")
flags.DEFINE_integer("test_size", 10000, "The size of test image pairs")
flags.DEFINE_integer("inner", 1, "The number of inner interation of training each network")
flags.DEFINE_integer("batch_size", 500, "The size of batch images [64]")
flags.DEFINE_string("conv_dim", '10,10,10', "The kernel numbers of conv layers ['20,40']")
flags.DEFINE_string("conv_kernel", '3,3,3', "The kernel dimensions of conv layers ['3,3']")
flags.DEFINE_integer("out_dim", 50, "The size of the output dimensions [50]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [mnist, xrmb]")
flags.DEFINE_boolean("classify", False, "If the data with the same label considered 'relevant'")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_float("alpha", 1, "Weight factor for negative samples")
flags.DEFINE_integer("NP_ratio", 5, "Ratio of Negtive samples to Positive samples")
flags.DEFINE_integer("feedback_epoch", 5, "Feedback epoch num")

FLAGS = flags.FLAGS


# --------------------------------
#             Main
# --------------------------------

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        bnn = BNN(
            sess,

            batch_size=FLAGS.batch_size,
            conv_dim=FLAGS.conv_dim,
            conv_kernel=FLAGS.conv_kernel,
            out_dim=FLAGS.out_dim,
            dataset_name=FLAGS.dataset,
            classify=FLAGS.classify,
            input_fname_pattern=FLAGS.input_fname_pattern,
            checkpoint_dir=FLAGS.checkpoint_dir,
            alpha=FLAGS.alpha,
            NP_ratio=FLAGS.NP_ratio,
            train_size=FLAGS.train_size,
            test_size=FLAGS.test_size
        )

        if FLAGS.train:
            bnn.train(FLAGS)

        if not bnn.load(FLAGS.checkpoint_dir)[0]:
            raise Exception("[!] Train a model first, then run test mode")


if __name__ == '__main__':
    tf.app.run()
