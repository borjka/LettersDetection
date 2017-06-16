import numpy as np
import tensorflow as tf
import os
from PIL import Image

check_path = os.getcwd()
if check_path.split('/')[1] == 'Users':
    path_to_letters = "/Users/borja/Stuff/imgs_of_letters/"
else:    
    path_to_letters = "/home/borja/imgs_of_letters/"

path_to_model = "trained_model/model"
images_for_letter = 10269
batch_size = 128
n_parts = 4
n_epochs = 5
part_of_GPU = 0.4
w = 32  
h = 32
main_dict = {}

class Model:
    X = None
    Y = None
    sample_size = 0
    N_letters = 52
    

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
        # print(files_list)

        arrs = []
        index = 0
        for f in files_list:
            main_dict[index] = f
            index += 1
            arrs.append(np.load(abs_path+f))
        # print(main_dict)

        Model.X = np.concatenate(arrs, axis=0)
        Model.sample_size = Model.X.shape[0]
        Model.N_letters = len(files_list)

        Model.Y = np.zeros((Model.sample_size, Model.N_letters))
        for i in range(Model.N_letters):
            Model.Y[i*images_for_letter:(i+1)*images_for_letter, i] = 1

        self.indices = [i for i in range(Model.sample_size)]
        np.random.shuffle(self.indices)
        print('<<<<<<<New part \"'+part+'\" of data loaded>>>>>>>')

        print(Model.X.shape)
        print(Model.Y.shape)


    def show_image(self, idx):
        if self.indices is None:
            print('Sample have not been downloaded. Firstly call \'load_sample()\'')
        else:
            if isinstance(idx, list):
                for i in idx:
                    Image.fromarray((self.X[i] * 255).astype('uint8'), 'L').show()
                    print(self.Y[i])
            elif isinstance(idx, int):
                Image.fromarray((self.X[idx] * 255).astype('uint8'), 'L').show()
                print(self.Y[idx])
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
        assert(Model.X is not None and Model.Y is not None)
        assert(Model.sample_size != 0)
    
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
            h1_after_dropout = tf.nn.dropout(h1, dropout)

        with tf.variable_scope('dense2'):
            W2 = tf.Variable(tf.truncated_normal([4096, Model.N_letters], stddev=0.01), name="W")
            b2 = tf.Variable(tf.constant(0.1, shape=[Model.N_letters]), name="b")
            logits = tf.matmul(h1_after_dropout, W2) + b2
            output = tf.nn.softmax(logits, name='outputs')

        with tf.variable_scope('train_stuff'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy, name='train_step')
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1), name='isCorrect')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')   
        
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=part_of_GPU)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint('trained_model/'))

        while (self.i+1) * batch_size < Model.sample_size:
            print('!!!!!BATCH '+str(self.i)+'!!!!!')
            X, Y = self.take_batch()
            feed_dict = {
                x: X,
                y: Y,
                dropout: 0.5,
                learning_rate: 0.0001
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
                if self.i % 100 == 0:
                    print(acc)
        saver.save(sess, path_to_model)


    def take_batch(self):
        if self.i % 10 == 0:
            print(str(self.i)+'/'+str(round(Model.sample_size / batch_size)))
        X_batch = Model.X[self.indices[self.i*batch_size : (self.i+1)*batch_size]].reshape(batch_size, h, w, 1)
        Y_batch = Model.Y[self.indices[self.i*batch_size : (self.i+1)*batch_size]]
        self.i += 1
        return X_batch, Y_batch

def test():
    my_model = Model()
    my_model.load_sample()
    my_model.build_graph(True)

def main():

    all_parts = ''
    for i in range(n_parts):
        all_parts += str(i)
    
    for ep in range(n_epochs):
        print("Epoch " + str(ep))
        for part in all_parts:
            my_model = None
            my_model = Model()
            my_model.load_sample(part)
            my_model.build_graph(True)


if __name__ == '__main__':
    test()


