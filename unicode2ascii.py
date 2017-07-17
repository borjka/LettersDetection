import numpy as np
import tensorflow as tf
import os
from PIL import Image
import letters_generator

path_to_letters = "imgs_of_letters/"
path_to_model = "trained_model/model"
all_letters = "abcdefghijklmnopqrstuvwxyzABDEFGHIJKLMNQRTVYZ"
little_letters = "abcdefghijklmnopqrstuvwxyz"
big_letters = "ABDEFGHIJKLMNQRTVYZ"
non_letters = "fgijkyhrFYR"
images_for_letter = 3528
batch_size = 128
n_parts = 8
n_epochs = 5
part_of_GPU = 0.4
w = 32
h = 32

class Model:
    X = None
    Y = None
    sample_size = 0
    N_letters = 45
    letter_to_number = {}
    number_to_letter = {}


    @classmethod
    def init_dict(cls):
        for i in range(len(little_letters)):
            cls.letter_to_number[little_letters[i]+'.npy'] = i
        for i in range(len(big_letters)):
            cls.letter_to_number['big_'+big_letters[i].lower()+'.npy'] = i + len(little_letters)
        cls.number_to_letter = {value:key for key,value in cls.letter_to_number.items()}


    def __init__(self):
        self.indices = None
        self.i = 0


    def load_sample(self, part='0'):
        '''Check that sample was loaded. If not will load it.
        Objects - Model.X
        Labels - Model.Y
        All instances in sample are shuffled.
        '''
        abs_path = path_to_letters+part+'/'
        files_list = os.listdir(abs_path)
        if files_list[0][0] == '.':
            del files_list[0]
        print(files_list)

        arrs = []
        Model.Y = None
        for f in files_list:
            Y_part = np.zeros((images_for_letter, Model.N_letters))
            if f.startswith('non'):
                Y_part[:, Model.N_letters-1] = 1
            else:
                Y_part[:, Model.letter_to_number[f]] = 1
            if Model.Y is None:
                Model.Y = Y_part
            else:
                Model.Y = np.vstack((Model.Y, Y_part))
            arrs.append(np.load(abs_path+f))

        Model.X = np.concatenate(arrs, axis=0)
        Model.sample_size = Model.X.shape[0]
        print(Model.Y.shape)
        print(Model.X.shape)

        self.indices = [i for i in range(Model.sample_size)]
        np.random.shuffle(self.indices)
        print('<<<<<<<New part \"'+part+'\" of data loaded>>>>>>>')


    def show_image(self, idx):
        if self.indices is None:
            print('Sample have not been downloaded. Firstly call \'load_sample()\'')
        else:
            if isinstance(idx, list):
                for i in idx:
                    Image.fromarray((self.X[i] * 255).astype('uint8'), 'L').show()
                    print(Model.number_to_letter[np.argmax(self.Y[i])])
            elif isinstance(idx, int):
                Image.fromarray((self.X[idx] * 255).astype('uint8'), 'L').show()
                print(Model.number_to_letter[np.argmax(self.Y[idx])])
            else:
                print('Wrong type of \'idx\'')

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def conv2d_nostrides(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def build_graph(self, checkCorrectness=False):
        '''Create Tensorflow graph and save it to "trained model" folder'''

        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=[None, h, w, 1], name='x')
        y = tf.placeholder(tf.float32, shape=[None, Model.N_letters], name='y')
        dropout = tf.placeholder(tf.float32, name='dropout')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        with tf.variable_scope('conv0'):
            W_conv0 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.01), name="W")
            b_conv0 = tf.Variable(tf.constant(0.1, shape=[32]), name="b")
            # (batch_size x 32 x 32 x 32)
            feature_map0 = tf.nn.relu(Model.conv2d_nostrides(x, W_conv0) + b_conv0)
            # tf.summary.histogram('W', W_conv0)
            # tf.summary.histogram('b', b_conv0)

        with tf.variable_scope('conv1'):
            W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 32, 32], stddev=0.01), name="W")
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name="b")
            # (batch_size x 16 x 16 x 32)
            feature_map1 = tf.nn.relu(Model.conv2d(feature_map0, W_conv1) + b_conv1)
            # tf.summary.histogram('W', W_conv1)
            # tf.summary.histogram('b', b_conv1)

        with tf.variable_scope('conv2'):
            W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01), name="W")
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name="b")
            # (batch_size x 8 x 8 x 64)
            feature_map2 = tf.nn.relu(Model.conv2d(feature_map1, W_conv2) + b_conv2)
            feature_map2_flatten = tf.reshape(feature_map2, [-1, 8 * 8 * 64])
            # tf.summary.histogram('W', W_conv2)
            # tf.summary.histogram('b', b_conv2)

        with tf.variable_scope('dense1'):
            W1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 4096], stddev=0.01), name="W")
            b1 = tf.Variable(tf.constant(0.1, shape=[4096]), name="b")
            h1 = tf.nn.relu(tf.matmul(feature_map2_flatten, W1) + b1)
            h1_after_dropout = tf.nn.dropout(h1, dropout)
            # tf.summary.histogram('W', W1)
            # tf.summary.histogram('b', W1)

        with tf.variable_scope('dense2'):
            W2 = tf.Variable(tf.truncated_normal([4096, Model.N_letters], stddev=0.01), name="W")
            b2 = tf.Variable(tf.constant(0.1, shape=[Model.N_letters]), name="b")
            logits = tf.matmul(h1_after_dropout, W2) + b2
            output = tf.nn.softmax(logits, name='outputs')
            # tf.summary.histogram('W', W2)
            # tf.summary.histogram('b', W2)


        with tf.variable_scope('train_stuff'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_step = optimizer.minimize(cross_entropy, name='train_step')

            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1), name='isCorrect')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
            # tf.summary.scalar('error', cross_entropy)
            # tf.summary.scalar('acc', accuracy)
        # merged = tf.summary.merge_all()

        saver = tf.train.Saver()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=part_of_GPU)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        # writer = tf.summary.FileWriter('TB', sess.graph)

        saver.restore(sess, tf.train.latest_checkpoint('trained_model/'))
        i = 0
        while True:
            X, Y = letters_generator.generate_batch()
            feed_dict = {
                x: X,
                y: Y,
                dropout: 0.5,
                learning_rate: 0.001
            }
            if checkCorrectness:
                acc, corr_pred, outp = sess.run([accuracy, correct_prediction, output],
                    feed_dict=feed_dict)
                # for i, item in enumerate(corr_pred):
                #     if not item:
                #         print(main_dict[np.argmax(outp[i])]+str(' : ')+main_dict[np.argmax(Y[i])])
                #         Image.fromarray((X[i, :, :, 0] * 255).astype('uint8'), 'L').show()
                print(acc)

            else:

                _, acc  = sess.run([train_step, accuracy], feed_dict=feed_dict)
                # writer.add_summary(summary)
                if i  % 10 == 0:
                    print(i, ' : ', acc)
            i += 1
            if i % 1000 == 0:
                print('saved')
                saver.save(sess, path_to_model)


    def take_batch(self):
        if self.i % 10 == 0:
            print(str(self.i)+'/'+str(round(Model.sample_size / batch_size)))
        X_batch = Model.X[self.indices[self.i*batch_size : (self.i+1)*batch_size]].reshape(batch_size, h, w, 1)
        Y_batch = Model.Y[self.indices[self.i*batch_size : (self.i+1)*batch_size]]
        self.i += 1
        return X_batch, Y_batch

def test():
    Model.init_dict()
    print(Model.letter_to_number)

def main():
    for ep in range(n_epochs):
        my_model = Model()
        my_model.build_graph()


if __name__ == '__main__':
    main()

