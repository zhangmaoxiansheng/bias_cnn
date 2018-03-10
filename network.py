#coding=utf-8
import numpy as np
from scipy import misc # feel free to use another image loader
import tensorflow as tf

total_size = 1060
batch_size = 100
num_epochs = 10000
trainable = 0
check_point_dir = "model_saved/model" #新版没有ckpt!!

#训练集输入处理
list_img = []
for i in range(0,1060):
    str = "/home/z/PycharmProjects/detectface/database/%d.jpg"%i
    list_img.append(str)

list_label = []
f = open("labels.txt")

while(1):
    line = f.readline()
    if len(line):
        line = line.replace('\n','')
        line = float(line)
        list_label.append(line)
        #print list_label[num]

    else:
        f.close()
        break
#测试集输入处理
test_label=[]
test_image = []
for i in range(100):
    str1 = "/home/z/PycharmProjects/detectface/test_database/tx%d.jpg"%i
    test_image.append(misc.imread(str1))
test_image = np.asarray(test_image)
f_t = open("/home/z/PycharmProjects/detectface/labels.txt")

while(1):
    line = f_t.readline()
    if len(line):
        line = line.replace('\n','')
        line = float(line)
        test_label.append(line)
    else:
        f_t.close()
        break

test_image = np.asarray(test_image)
test_label = np.asarray(test_label)
test_label = np.reshape(test_label,(100,1))

#训练集生成batch
def create_batches(batch_size):
    images = []
    for img in list_img:
        images.append(misc.imread(img))
    images = np.asarray(images)#统一转化为np的array

    labels = np.asarray(list_label)
    labels = np.reshape(labels,(1060,1))

    while (True):
        for i in range(0, total_size, batch_size):
            yield (images[i:i + batch_size], labels[i:i + batch_size])#这是一个生成器，yield返回一个生成器对象（有return功能）
#input:

imgs = tf.placeholder(tf.float32,shape=[None,60,80,3])
lbls = tf.placeholder(tf.float32, shape = [None,1])
keep_prob = tf.placeholder(tf.float32)

#weight,bias and cnn layer

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial,name)

def bias_variable(shape,name):
    initial = tf.constant(0.01,shape=shape)
    return tf.Variable(initial , name)

def conv2d(x,W,stride):
    return tf.nn.conv2d(x , W , strides=[1,stride,stride,1] , padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x , ksize=[1,2,2,1] , strides = [1,2,2,1] , padding = "SAME")

with tf.Session() as sess:
    # conv network weight
    W_conv1 = weight_variable([5,5,3,32],"W_conv1") #卷积核边长为5，输入层深度为3，输出层深度为32
    b_conv1 = bias_variable([32],"b_conv1") #偏置

    W_conv2 = weight_variable([3,3,32,64],"W_conv2") #与上层深度应一致，池化无法改变深度
    b_conv2 = bias_variable([64],"b_conv2")

    W_conv3 = weight_variable([2,2,64,64],"W_conv3")
    b_conv3 = bias_variable([64],"b_conv3")

    #conv network
    h_conv1 = tf.nn.relu(conv2d(imgs,W_conv1,2) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3,2) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #将张量抽成一维向量
    pool_shape = h_pool3.get_shape().as_list()
    #注意这里的pool_shape[0]是一个batch中的数量
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    h_pool3_flat = tf.reshape(h_pool3,[-1,nodes])

    #full connected network
    W_fc1 = weight_variable([nodes,256],"W_fc1")
    b_fc1 = bias_variable([256],"b_fc1")

    W_fc2 = weight_variable([256,128],"W_fc2")
    b_fc2 = bias_variable([128],"b_fc2")

    W_fc3 = weight_variable([128,1],"W_fc3")
    b_fc3 = bias_variable([1],"b_fc3")


    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)

    result = tf.matmul(h_fc2_drop,W_fc3) + b_fc3

    #loss
    loss = tf.reduce_mean(tf.square(result - lbls))

    #record the loss
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter("logs/test", sess.graph)
    train_writer = tf.summary.FileWriter("logs/train" ,sess.graph)

    #saver
    saver = tf.train.Saver()


    #train the net
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
    batch_generator = create_batches(batch_size)
    sess.run(tf.global_variables_initializer())

    if trainable:
        for i in range(num_epochs):
            images,labels = batch_generator.next()
            _ , train_loss = sess.run([train_step,loss],feed_dict = {imgs:images,lbls:labels,keep_prob:0.5})
            #训练测试
            if i%100 == 0:

                print "train_loss is %f"%train_loss
                test_loss = sess.run(loss,feed_dict = {imgs:test_image,lbls:test_label,keep_prob:1})
                print "test_loss is %f"%test_loss

                test_loss_record = sess.run(merged, feed_dict= {imgs:test_image,lbls:test_label,keep_prob:1})  #测试时不需要用dropout，keep=1即可
                train_loss_record = sess.run(merged , feed_dict={imgs:images,lbls:labels,keep_prob:1})

                test_writer.add_summary(test_loss_record,i)
                train_writer.add_summary(train_loss_record,i)

                saver.save(sess,check_point_dir)

    else:
        saver.restore(sess,'./model_saved/model')
        list_use = []
        image_use = misc.imread('/home/z/PycharmProjects/detectface/database/1.jpg')
        list_use.append(image_use)
        list_use = np.asarray(list_use)
        tf.reshape(list_use,[1,60,80,3])#将图片reshape
        print(sess.run(result,feed_dict={imgs:list_use,keep_prob:1}))



