import tensorflow as tf
import numpy as np
import letters_generator
import homography

class Model:

    def __init__(self,
                 img_h=32,
                 img_w=32,
                 channels=1,
                 path_to_model="trained_model/"
                ):
        self.one_hot_len = letters_generator.N_symbols+1
        self.training_iteration = 0
        self.accuracy = 0
        self.learning_rate = tf.placeholder(tf.float32)
        self.dropout = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32, shape=[None, img_h, img_w, channels])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.one_hot_len])
        self.transformation_matrix = tf.placeholder(tf.float32, shape=[8])
        self.train_step = None
        self.sess = None
        self.saver = None
        self.processed_X = None
        self._Y = None
        self.path_to_model = path_to_model



    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


    @staticmethod
    def conv2d_nostrides(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def build_graph(self, isToProcess=True):
        if isToProcess:
            self.processed_X = tf.contrib.image.transform(self.X, self.transformation_matrix)
        else:
            self.processed_X = self.X

        with tf.variable_scope('conv0'):
            W_conv0 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.01), name="W")
            b_conv0 = tf.Variable(tf.constant(0.1, shape=[32]), name="b")
            # (batch_size x 32 x 32 x 32)
            feature_map0 = tf.nn.relu(Model.conv2d_nostrides(self.processed_X, W_conv0) + b_conv0)

        with tf.variable_scope('conv1'):
            W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 32, 32], stddev=0.01), name="W")
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name="b")
            # (batch_size x 16 x 16 x 32)
            feature_map1 = tf.nn.relu(Model.conv2d(feature_map0, W_conv1) + b_conv1)

        with tf.variable_scope('conv2'):
            W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01), name="W")
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name="b")
            # (batch_size x 8 x 8 x 64)
            feature_map2 = tf.nn.relu(Model.conv2d(feature_map1, W_conv2) + b_conv2)
            feature_map2_flatten = tf.reshape(feature_map2, [-1, 8 * 8 * 64])

        with tf.variable_scope('dense1'):
            W1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 4096], stddev=0.01), name="W")
            b1 = tf.Variable(tf.constant(0.1, shape=[4096]), name="b")
            h1 = tf.nn.relu(tf.matmul(feature_map2_flatten, W1) + b1)
            h1_after_dropout = tf.nn.dropout(h1, self.dropout)

        with tf.variable_scope('dense2'):
            W2 = tf.Variable(tf.truncated_normal([4096, self.one_hot_len], stddev=0.01), name="W")
            b2 = tf.Variable(tf.constant(0.1, shape=[self.one_hot_len]), name="b")
            logits = tf.matmul(h1_after_dropout, W2) + b2
            self.Y_ = tf.nn.softmax(logits, name='outputs')

        with tf.variable_scope('train_stuff'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=logits))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_step = optimizer.minimize(cross_entropy, name='train_step')

            correct_prediction = tf.equal(tf.argmax(self.Y_, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def save_model(self):
        if self.sess is not None:
            self.saver.save(self.sess, self.path_to_model)
        else:
            print('Nothing to save. First train something')


    def restore_model(self):
        if self.sess is not None:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.path_to_model))
        else:
            print('Nowhere to restore. First build the graph.')


    def train_model(self,
                    isToRestore=False,
                    learning_rate=0.001,
                    dropout=0.5,
                    N_iter=1000):

        if isToRestore:
            self.restore_model()

        for i in range(N_iter):
            X_batch, Y_batch = letters_generator.generate_batch()
            feed_dict = {
                self.X: X_batch,
                self.Y: Y_batch,
                self.dropout: dropout,
                self.transformation_matrix: homography.generate_transformation_matrix(),
                self.learning_rate: learning_rate
            }

            _, acc = self.sess.run([self.train_step, self.accuracy], feed_dict=feed_dict)
            if i % 20 == 0:
                print('Iteration {0} / {1}'.format(i, N_iter))
                print('Accuracy: {0}'.format(acc))
        self.save_model()


def main():
    my_model = Model()
    my_model.build_graph()
    my_model.train_model()


if __name__ == '__main__':
    main()
