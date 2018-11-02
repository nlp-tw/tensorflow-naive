import tensorflow as tf
import numpy as np

matA = tf.placeholder(tf.int32, [2,2], name="matA")
matB = tf.placeholder(tf.int32, [2,2], name="matB")

multi = matA * matB
matmul = tf.matmul(matA, matB)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(multi, feed_dict={matA: np.array([[1,2], [3,4]]), matB: np.array([[10, 20], [30, 40]])}))
print(sess.run(matmul, feed_dict={matA: np.array([[1,2], [3,4]]), matB: np.array([[10, 20], [30, 40]])}))

sess.close()
