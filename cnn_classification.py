#coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_DATA',one_hot=True)


x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)    

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2X2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  

x_image = tf.reshape(x,[-1,28,28,1])

W_conv1 = weight_variable([5,5,1,32])#pacth 5x5,input chanels=1,output chanels=32    
b_conv1 = bias_variable([32])#对于每个output chanels 都有对应的一个 bias
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2X2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])#pacth 5x5,input chanels=1,output chanels=32    
b_conv2 = bias_variable([64])#对于每个output chanels 都有对应的一个 bias
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2X2(h_conv2)

#full connected layer
W_fc1 = weight_variable([7*7*64,1024])#pooling时：28 - 14 - 7
b_fc1 = bias_variable([1024])
#整平化
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

#减少过拟合，加入dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#output layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)


# with tf.Session() as sess:
#     cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv)))
#     train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # sess.run(tf.global_variables_initializer())
    # for i in range(2000):
    #     batch = mnist.train.next_batch(50)
    #     if i%100 == 0:
    #         train_accuracy = accuracy.eval(feed_dict={
    #             x:batch[0], y_: batch[1], keep_prob: 1.0})
    #         print( "step %d, training accuracy %g" %(i, train_accuracy))
    #     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # print( "test accuracy %g" %accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
def compute_accuracy(ac_x,ac_y):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    result = sess.run(accuracy,feed_dict={x:ac_x,y_:ac_y,keep_prob:1.0})
    return result

for i in range(10001):
    batch = mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})    
    if i%100 == 0:
        print('step %d accuracy %g' %(i,compute_accuracy(mnist.test.images,mnist.test.labels)))

sess.close()        