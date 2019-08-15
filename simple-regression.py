import tensorflow.compat.v1 as tf
import numpy as np

# Weight and bias variables initialized to 0
W = tf.Variable([[0.0]])
b = tf.Variable(0.0)

X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])

predictions = tf.add(tf.matmul(X, W), b)
loss = tf.reduce_mean(tf.square(predictions - Y))
model = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# Sample imput data. y = 3.x + 4
train_x = np.array([[1.0], [2.0], [3.0], [4.0]])
train_y = train_x * 3.0 + 4.0

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for train_step in range(40001):
        sess.run(model, {X:train_x, Y:train_y})

        # Print training progress
        if train_step % 2000 == 0:
            error_rate = sess.run(loss, {X:train_x, Y:train_y})
    
            print("Step:", train_step, 
                "W:", sess.run(W), 
                "b:", sess.run(b), 
                "Loss:", error_rate)
            if error_rate < 0.0001:
                saver.save(sess, "./model.ckpt")
                break

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # Load the weights and biases
    saver.restore(sess, "./model.ckpt")

    # Validate the model with data not used in training
    x_unseen = np.array([[6.0], [7.0], [8.0]])
    y_expected = x_unseen * 3.0 + 4.0
    print("Predections:", sess.run(predictions, {X:x_unseen}))
    print("Expected:", y_expected)
