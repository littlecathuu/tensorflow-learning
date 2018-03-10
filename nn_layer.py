#coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def nn_layer(inputs,in_size,out_size,activation_func=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))#正态分布的权重矩阵
    biases = tf.Variable(tf.zeros([1,out_size])+0.05)#偏置项
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_func is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_func(Wx_plus_b)
    return outputs

#生成训练数据
x_data = np.linspace(-1,1,300)[:,np.newaxis]#np.newaxis增加维度
noise = np.random.normal(0,0.05,x_data.shape)#标准正态分布，均值μ=0，标准差σ=1
y_data = np.square(x_data) - 0.1 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

lay1 = nn_layer(xs,1,10,activation_func=tf.nn.relu)
prediction = nn_layer(lay1,10,1,activation_func=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data,y_data)
# plt.ion()
# plt.show()

#training
# for i in range(1000):
#     sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
#     if i%50 == 0:
#         print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        # try:
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass
        # prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        # lines = ax.plot(x_data,prediction_value,'r+',lw=5)
        # plt.pause(0.1)
        # sess.close()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        l = sess.run(loss,feed_dict={xs:x_data,ys:y_data})
        if i%50 == 0:
            print(i,l)