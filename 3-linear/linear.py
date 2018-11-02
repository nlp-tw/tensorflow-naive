import tensorflow as tf

# Model Parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Inputs and outputs
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# Loss
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

print(sess.run(linear_model, {x: [1,2,3,4], y: [0,-1,-2,-3]}))
print(sess.run(squared_delta, {x: [1,2,3,4], y: [0,-1,-2,-3]}))
print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))


