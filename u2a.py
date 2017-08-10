import tensorflow as tf
import numpy as np
import letters_generator
import homography
import math
import os
import time
from PIL import Image

class Model:

    def __init__(self,
                 img_h=32,
                 img_w=32,
                 channels=1,
                 path_to_model="trained_model/"
                ):
        self.one_hot_len = letters_generator.N_symbols
        self.training_iteration = 0
        self.accuracy = 0
        self.learning_rate = tf.placeholder(tf.float32)
        self.dropout1 = tf.placeholder(tf.float32)
        self.dropout2 = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32, shape=[None, img_h, img_w, channels])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.one_hot_len])
        self.transformation_matrix = tf.placeholder(tf.float32, shape=[8])
        self.train_step = None
        self.sess = None
        self.saver = None
        self.processed_X = None
        self._Y = None
        self.var_to_save = None
        self.path_to_model = path_to_model
        self.logits_to_check = None



    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


    @staticmethod
    def conv2d_nostrides(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pooling(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def build_graph(self, isToProcess=True):
        if isToProcess:
            self.processed_X = tf.contrib.image.transform(self.X, self.transformation_matrix)
        else:
            self.processed_X = self.X

        with tf.variable_scope('conv0'):
            W_conv0 = tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.01), name="W")
            b_conv0 = tf.Variable(tf.constant(0.01, shape=[64]), name="b")
            feature_map0_before_pool = tf.nn.relu(Model.conv2d_nostrides(self.processed_X, W_conv0) + b_conv0)
            feature_map0 = Model.max_pooling(feature_map0_before_pool)
            # output is (batch_size x 32 x 32 x 64)

        with tf.variable_scope('conv1'):
            W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 64, 128 ], stddev=0.01), name="W")
            b_conv1 = tf.Variable(tf.constant(0.01, shape=[128]), name="b")
            feature_map1 = tf.nn.relu(Model.conv2d_nostrides(feature_map0, W_conv1) + b_conv1)
            # output is (batch_size x 16 x 16 x 128)

        with tf.variable_scope('conv2'):
            W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.01), name="W")
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[128]), name="b")
            feature_map2_before_pool = tf.nn.relu(Model.conv2d_nostrides(feature_map1, W_conv2) + b_conv2)
            feature_map2 = Model.max_pooling(feature_map2_before_pool)
            feature_map2_flatten_before_dropout = tf.reshape(feature_map2, [-1, 8 * 8 * 128])
            feature_map2_flatten = tf.nn.dropout(feature_map2_flatten_before_dropout, self.dropout1)
            # output is (batch_size x 8 x 8 x 128)


        with tf.variable_scope('dense1'):
            W1 = tf.Variable(tf.truncated_normal([8 * 8 * 128, 4096], stddev=0.01), name="W")
            b1 = tf.Variable(tf.constant(0.01, shape=[4096]), name="b")
            h1 = tf.nn.relu(tf.matmul(feature_map2_flatten, W1) + b1)
            h1_after_dropout = tf.nn.dropout(h1, self.dropout2)

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
        self.var_to_save = W_conv2
        self.logits_to_check = logits


    def save_model(self):
        if self.sess is not None:
            self.saver.save(self.sess, self.path_to_model + 'my_model')
        else:
            print('Nothing to save. First train something')


    def restore_model(self):
        if self.sess is not None:
            if os.path.exists(self.path_to_model + 'checkpoint'):
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.path_to_model))
        else:
            print('Nowhere to restore. First build the graph.')


    def interesting_weights(self):
        self.restore_model()
        return self.sess.run(self.var_to_save)


    def check_top3_logits(self, img_pxls):
        self.restore_model()

        if len(img_pxls.shape) !=  4:
            pxls = np.reshape(img_pxls, (1, img_pxls.shape[0], img_pxls.shape[1], 1))
        else:
            pxls = img_pxls

        feed_dict = {
            self.X: pxls,
            self.dropout1: 1,
            self.dropout2: 1,
        }

        logits = self.sess.run([self.logits_to_check], feed_dict=feed_dict)[0][0]
        args_of_max = logits.argsort()[-3:][::-1]
        answ = []
        # for arg_i in args_of_max:
        arg_i = args_of_max[0]
        if arg_i == self.one_hot_len - 1:
            # print('non_letter')
            lbl = 'nl'
            # answ.append('nl')
        else:
            # print(letters_generator.all_symbols[arg_i], " : ", logits[arg_i])
            lbl = letters_generator.all_symbols[arg_i]
            # answ.append(letters_generator.all_symbols[arg_i])
        img = Image.fromarray((pxls[0, :, :, 0] * 255).astype('uint8'), 'L')
        _id = np.random.randint(1000000)
        img.save('u2a/{0}_{1}.png'.format(lbl, _id))
        # Image.fromarray((pxls[0, :, :, 0] * 255).astype('uint8'), 'L').show()
        return answ


    def train_model(self,
                    isToRestore=True,
                    learning_rate=0.0001,
                    dropout=0.5,
                    N_iter=10000):

        if isToRestore:
            self.restore_model()

        for i in range(N_iter):
            X_batch, Y_batch = letters_generator.generate_batch()
            feed_dict = {
                self.X: X_batch,
                self.Y: Y_batch,
                self.dropout1: dropout,
                self.dropout2: dropout,
                self.transformation_matrix: homography.generate_transformation_matrix(),
                self.learning_rate: learning_rate
            }

            _, acc = self.sess.run([self.train_step, self.accuracy], feed_dict=feed_dict)
            if i % 20 == 0:
                print('Iteration {0} / {1}'.format(i, N_iter))
                print('Accuracy: {0}'.format(acc))
            if i % 2000 == 0 or i == N_iter - 1:
                self.save_model()



    def classify_one_image(self, img_pxls):
        self.restore_model()

        if len(img_pxls.shape) !=  4:
            pxls = np.reshape(img_pxls, (1, img_pxls.shape[0], img_pxls.shape[1], 1))
        else:
            pxls = img_pxls

        feed_dict = {
            self.X: pxls,
            self.dropout: 1,
        }

        answ  = self.sess.run([self.Y_], feed_dict=feed_dict)
        letter_index = np.argmax(answ[0])
        return letters_generator.all_symbols[letter_index]


    def show_me_sample(self, andSave=True):
        self.restore_model()

        X_batch, Y_batch = letters_generator.generate_batch()
        feed_dict = {
            self.X: X_batch,
            self.Y: Y_batch,
            # self.dropout1: dropout,
            # self.dropout2: dropout,
            self.transformation_matrix: homography.generate_transformation_matrix(),
            # self.learning_rate: learning_rate
        }

        pxls = self.sess.run([self.processed_X], feed_dict=feed_dict)[0]

        N_imgs = 64
        for i in range(N_imgs):
            if andSave:
                img = Image.fromarray((pxls[i, :, :, 0] * 255).astype('uint8'), 'L')
                img.save('answers/{0}.png'.format(i))




    def classify_batch_of_images(self, imgs_pxls, andSave=False):
        self.restore_model()

        N_imgs = imgs_pxls.shape[0]
        if len(imgs_pxls.shape) !=  4:
            pxls = np.reshape(imgs_pxls, (N_imgs, imgs_pxls.shape[1], imgs_pxls.shape[2], 1))
        else:
            pxls = imgs_pxls

        feed_dict = {
            self.X: pxls,
            self.dropout1: 1,
            self.dropout2: 1,
        }

        answ = self.sess.run([self.Y_], feed_dict=feed_dict)[0]
        labels = []
        for i in range(N_imgs):
            letter_index = np.argmax(answ[i])
            if letter_index == self.one_hot_len - 1:
                labels.append('non_letter')
            else:
                labels.append(letters_generator.all_symbols[letter_index])
            if andSave:
                img = Image.fromarray((pxls[i, :, :, 0] * 255).astype('uint8'), 'L')
                img.save('answers/{0}_{1}.png'.format(labels[i], i))
        return labels


def save_weights_for_visualization(weights):
    arr_weights = []
    w = []
    for i in range(32):
        arr_weights.append(weights[:, :, 0, i])
        min_val = np.amin(arr_weights[i])
        max_val = np.amax(arr_weights[i])
        w.append((arr_weights[i] - min_val) / (max_val - min_val))

        img = Image.fromarray((w[i] * 255).astype('uint8'), 'L')
        img.save('weights/'+str(i)+'.png')


def main():
    my_model = Model()
    my_model.build_graph(isToProcess=True)
    # save_weights_for_visualization(my_model.interesting_weights())
    # my_model.show_me_sample()
    my_model.train_model(N_iter=10000)

    #### CLASSIFY UNICODE #####
    # basic_path = 'all_unicode/'
    # all_images = os.listdir(basic_path)
    # symbols = []
    # for img_name in all_images:
        # if img_name[0] == '.':
            # continue
        # img = Image.open(basic_path+img_name)
        # symbols.append(np.array(img) / 255)
    # symbols = np.array(symbols)
    # my_model.classify_batch_of_images(symbols, andSave=True)

    #### CHECK LOGITS #####
    # basic_path = 'all_unicode/'
    # all_images = os.listdir(basic_path)
    # if all_images[0][0] == '.':
        # del all_images[0]
    # for img_name in all_images:
        # img = Image.open(basic_path+img_name)
        # img = np.array(img) / 255
        # my_model.check_top3_logits(img)


if __name__ == '__main__':
    main()
