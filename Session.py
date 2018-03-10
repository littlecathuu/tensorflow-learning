#coding=utf-8
import tensorflow as tf

matrix1 = tf.constant([[2,3]])
matrix2 = tf.constant([[4],[5]])

dot_op = tf.matmul(matrix1,matrix2)
#由tensor和operation组成的graph需在session中打开
sess = tf.Session()

result = sess.run(dot_op)

print(result)
#操作完成后不要忘了关闭session
sess.close()

#一种隐式关闭session

# with tf.Session as sess:
#     result = sess.run(dot_op)
#     print(result)