import tensorflow.compat.v1 as tf
import numpy as np

def generate_training_data(batch_size):
    x = np.zeros([batch_size, 3])
    y = np.zeros([batch_size, 1])

    for index in range(0, batch_size):
        x[index, 0] = (index * 3 + 1.0) * 0.001
        x[index, 1] = (index * 3 + 2.0) * 0.001
        x[index, 2] = (index * 3 + 3.0) * 0.001
        y[index, 0] = x[index, 0] + 3.7 * x[index, 1] + 5.9 * x[index, 2]
    
    return (x, y)

W = tf.Variable(tf.truncated_normal([3, 1], stddev=0.001))
b = tf.Variable(tf.truncated_normal([1], stddev=0.001))

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 1])

labels = tf.add(tf.matmul(X, W), b)
loss = tf.reduce_mean(tf.square(labels - Y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

x, y = generate_training_data(30)

print(x)
print(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for train_step in range(40001):
        sess.run(train, {X:x, Y:y})

        error_rate = sess.run(loss, {X:x, Y:y})

        if train_step % 2000 == 0:
            print("Step:", train_step, 
                "W:", sess.run(W), 
                "b:", sess.run(b), 
                "Loss:", error_rate)
            if error_rate < 0.0001:
                break

    # x_unseen = [[6.0, 7.0, 8.0]]
    # print("Predections:", sess.run(labels, {X:x_unseen}))
