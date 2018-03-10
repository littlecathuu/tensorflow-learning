#coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_DATA',one_hot=True)

def nn_layer(inputs,in_size,out_size,activation_func=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))#正态分布的权重矩阵
    biases = tf.Variable(tf.zeros([1,out_size])+0.05)#偏置项
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_func is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_func(Wx_plus_b)
    return outputs


xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

prediction = nn_layer(xs,784,10,activation_func=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),#axis=1
reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
def compute_accuracy(v_xs,v_ys):
    # y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(v_ys,1))
    accuracy =tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(10001):
    batch_xs,batch_ys = mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%100 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
