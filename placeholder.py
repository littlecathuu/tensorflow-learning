#coding=utf-8
import tensorflow as tf

x_1 = tf.placeholder(dtype=tf.float32,shape=None)
y_1 = tf.placeholder(dtype=tf.float32,shape=None)
z_1 = x_1 + y_1

x_2 = tf.placeholder(tf.float32,shape=[2,1])
y_2 = tf.placeholder(tf.float32,shape=[1,2])
z_2 =tf.matmul(x_2,y_2)

with tf.Session() as sess:
    #only one operation
    value_1 = sess.run(z_1,feed_dict={x_1:2,y_1:3})
    #multiple operations
    value_1,value_2 = sess.run([z_1,z_2],feed_dict={x_1:4,y_1:5,x_2:[[3],[5]],y_2:[[2,4]]})
    print(value_1)
    print(value_2)