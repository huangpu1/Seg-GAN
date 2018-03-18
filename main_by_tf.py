import argparse
import DeconvNet
import time
import numpy as np
import datetime
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_record', help="training tfrecord file", default="tfrecords/pascalvoc2012.tfrecords")
    parser.add_argument('--train_dir', help="where to log training", default="train_log")
    parser.add_argument('--batch_size', help="batch size", type=int, default=10)
    parser.add_argument('--num_epochs', help="number of epochs.", type=int, default=50)
    parser.add_argument('--lr', help="learning rate", type=float, default=1e-6)
    args = parser.parse_args()

    trn_images_batch, trn_segmentations_batch = input_pipeline(
        args.train_record,
        args.batch_size,
        args.num_epochs)

    deconvnet = DeconvNet(trn_images_batch, trn_segmentations_batch, use_cpu=False)

    logits = deconvnet.logits

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(tf.reshape(deconvnet.y, [-1]), tf.int64), logits=logits, name = 'x_entropy')

    loss_mean = tf.reduce_mean(cross_entropy, name='x_entropy_mean')

    train_step = tf.train.AdamOptimizer(args.lr).minimize(loss_mean)

    summary_op = tf.summary.merge_all()  # v0.12

    # init = tf.initialize_all_variables()
    # init_locals = tf.initialize_local_variables()

    init_global = tf.global_variables_initializer()  # v0.12
    init_locals = tf.local_variables_initializer()  # v0.12

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as sess:

        sess.run([init_global, init_locals])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # summary_writer = tf.train.SummaryWriter(args.train_dir, sess.graph)
        summary_writer = tf.summary.FileWriter(args.train_dir, sess.graph)  # v0.12
        # training_summary = tf.scalar_summary("loss", loss_mean)
        training_summary = tf.summary.scalar("loss", loss_mean)  # v0.12

        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                _, loss_val, train_sum = sess.run([train_step, loss_mean, training_summary])
                elapsed = time.time() - start_time
                summary_writer.add_summary(train_sum, step)
                # print sess.run(deconvnet.prediction)

                assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

                step += 1

                if step % 1 == 0:
                    num_examples_per_step = args.batch_size
                    examples_per_sec = num_examples_per_step / elapsed
                    sec_per_batch = float(elapsed)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_val,
                                        examples_per_sec, sec_per_batch))


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)