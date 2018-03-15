import tensorflow as tf
class DeconvNet:


    def get_data(self):
        """
        Download and unpack VOC data if data folder not exists data
        :return:
        """
        import os, wget, tarfile
        if os.listdir("data") == ['.gitignore']:
            filenames = ['VOC_OBJECT.tar.gz', 'VOC2012_SEG_AUG', 'stage_1_train_imgset.tar.gz', 'stage_2_train_imgset.tar.gz']
            url = 'http://cvlab.postech.ac.kr/research/deconvnet/data'
            for filename in filenames:
                wget.download(url + filename, out = os.path.join('data', filename))
                tar = tarfile.open(os.path.join('data', filename))
                tar.extract(path='data')
                tar.close()

                os.remove(os.path.join('data', filename))
        return

    def build(self, use_cpu=False):
        """
        build the deconvnet
        :param use_cpu: use cpu to test or train
        :return:
        """

        if use_cpu:
            device = '/cpu:0'
        else:
            device = '/gpu:0'

        with tf.device(device):
            self.x = tf.placeholder(tf.float32, shape=(1, None, None, 3))
            self.y = tf.placeholder(tf.int64, shape=(1, None, None))

            expected = tf.expand_dims(self.y, -1)
            self.rate = tf.placeholder(tf.float32, shape=[])



    def weight_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
        W = self.weight_variable(W_shape)
        b= self.bias_variable([b_shape])
        return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1,1,1,1], padding=padding) + b)

    def pool_layer(self, x):
        with tf.device('/gpu:0'):
            return tf.nn.max_pool_with_argmax(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def deconv_layer(self, x, W_Shape, b_shape, name, padding='SAME'):
        W = self.weight_variable(W_Shape)
        b = self.bias_variable([b_shape])

        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W_Shape[2]])
        return tf.nn.conv2d_transpose(x,W, out_shape, [1,1,1,1], padding)

    def unravel_argmax(self, argmax, shape):
        output_list=[]
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3])// shape[3])
        return  tf.stack(output_list)

    def unpool_layer2x2(self, x, raveled_argmax, out_shape):
        argmax = self.unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
        out_shape tf.zeros([out_shape[1], out_shape[2],out_shape[3]])