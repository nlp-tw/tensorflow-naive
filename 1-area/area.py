import tensorflow as tf

width = tf.placeholder(tf.int32, name="w")
height = tf.placeholder(tf.int32, name="h")
area = tf.multiply(width, height, name="area")
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
res = sess.run(area, feed_dict={width:8, height: 8})
print(res)
sess.close()
