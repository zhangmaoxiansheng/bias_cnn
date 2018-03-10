#coding=utf-8
import tensorflow as tf
import numpy as np
from scipy import misc

class bias_cnn:
    def __init__(self,
                 test_num = 1,
                 trainable=1,
                 num_batch=100,
                 eposide=10000,
                 img_dir="database",
                 label_dir = "labels.txt",
                 save_path = "model_saved/model",
                 test_path = "database"):
        self.num_batch = num_batch
        self.epo = eposide
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.save_path = save_path
        self.test_path = test_path
        self.test_num = test_num
        self.get_label()
        self.get_img()
        self.build_net()
        if trainable == 1:
            self.train_net()
        else:
            self.use_net()

    def get_label(self):
        self.labels = []
        f = open(self.label_dir)
        self.total_num = 0
        while(1):
            line = f.readline()
            if len(line):
                self.total_num += 1
                line = line.replace('\n', '')
                line = float(line)
                self.labels.append(line)
            else:
                f.close()
                break
        self.labels = np.asarray(self.labels)
        self.labels = np.reshape(self.labels,(self.total_num,1))

    def get_img(self):
        self.images = []
        for i in range(0, self.total_num):
            str1 = '%s/%d.jpg' % (self.img_dir, i)
            # list_img.append(str1)
            self.images.append(misc.imread(str1))

        self.images = np.asarray(self.images)  # 统一转化为np的array

    def create_batch(self):
        while (True):
            for i in range(0, self.total_num, self.num_batch):
                yield (self.images[i:i + self.num_batch], self.labels[i:i + self.num_batch])  # 这是一个生成器，yield返回一个生成器对象（有return功能）

    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name)

    @staticmethod
    def bias_variable(shape,name):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial,name)

    @staticmethod
    def conv2d(x,W,stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def build_net(self):
        #input
        self.imgs = tf.placeholder(tf.float32, shape=[None, 60, 80, 3])
        self.lbls = tf.placeholder(tf.float32, shape=[None, 1])
        self.keep_prob = tf.placeholder(tf.float32)
        #cnn
        W_conv1 = self.weight_variable([5, 5, 3, 32], "W_conv1")  # 卷积核边长为5，输入层深度为3，输出层深度为32
        b_conv1 = self.bias_variable([32], "b_conv1")  # 偏置

        W_conv2 = self.weight_variable([3, 3, 32, 64], "W_conv2")  # 与上层深度应一致，池化无法改变深度
        b_conv2 = self.bias_variable([64], "b_conv2")

        W_conv3 = self.weight_variable([2, 2, 64, 64], "W_conv3")
        b_conv3 = self.bias_variable([64], "b_conv3")

        # conv network
        h_conv1 = tf.nn.relu(self.conv2d(self.imgs, W_conv1, 2) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3, 2) + b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)

        # 将张量抽成一维向量
        pool_shape = h_pool3.get_shape().as_list()
        # 注意这里的pool_shape[0]是一个batch中的数量
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        h_pool3_flat = tf.reshape(h_pool3, [-1, nodes])

        # full connected network
        W_fc1 = self.weight_variable([nodes, 256], "W_fc1")
        b_fc1 = self.bias_variable([256], "b_fc1")

        W_fc2 = self.weight_variable([256, 128], "W_fc2")
        b_fc2 = self.bias_variable([128], "b_fc2")

        W_fc3 = self.weight_variable([128, 1], "W_fc3")
        b_fc3 = self.bias_variable([1], "b_fc3")

        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

        self.result = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    def train_net(self):
        self.loss = tf.reduce_mean(tf.square(self.result - self.lbls))
        train_step = tf.train.AdamOptimizer(1e-5).minimize(self.loss)
        batch_generator = self.create_batch()
        # record
        tf.summary.scalar('loss', self.loss)
        merged = tf.summary.merge_all()
        # saver
        self.saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter("logs/train", sess.graph)

            for i in range(self.epo):
                img_in, label_in = batch_generator.next()
                _, train_loss = sess.run([train_step, self.loss],
                                             feed_dict={self.imgs: img_in, self.lbls: label_in, self.keep_prob: 0.5})
                if i % 100 == 0:
                    print "train_loss is %f" % train_loss
                    train_loss_record = sess.run(merged, feed_dict={self.imgs: img_in, self.lbls: label_in, self.keep_prob: 1})
                    train_writer.add_summary(train_loss_record, i)
                    self.saver.save(sess, self.save_path)

    def use_net(self):
        with tf.Session() as sess:
            self.saver.restore(sess,self.save_path)
            list_use = []
            str2 = '%s/%d.jpg' %(self.test_path,self.test_num)
            image_use = misc.imread(str2)#测试路径
            list_use.append(image_use)
            list_use = np.asarray(list_use)
            tf.reshape(list_use, [1, 60, 80, 3])  # 将图片reshape
            print(sess.run(self.result, feed_dict={self.imgs: list_use, self.keep_prob: 1}))




if __name__ == "__main__":
    pre_cnn = bias_cnn()











