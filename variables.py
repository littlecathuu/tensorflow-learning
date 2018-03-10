#coding=utf-8
import tensorflow as tf

var = tf.Variable(0)

add_op = tf.add(var,1)
update_op = tf.assign(var,add_op)#该函数将add_op -> var，返回var

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#initialize_all_variables被弃用
   
    for i in range(10):
        sess.run(update_op)
        print(i,sess.run(var))