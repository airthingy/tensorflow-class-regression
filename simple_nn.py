import tensorflow.compat.v1 as tf
import numpy as np

def hidden_layer(input_tensor, num_neurons):
    num_input_features = input_tensor.get_shape()[1].value
    W = tf.Variable(tf.truncated_normal([num_input_features, num_neurons], stddev=0.1))
    b = tf.Variable(tf.zeros([num_neurons]))

    return tf.nn.relu(tf.matmul(input_tensor, W) + b)

def readout_layer(input_tensor):
    num_input_features = input_tensor.get_shape()[1].value
    W = tf.Variable(tf.truncated_normal([num_input_features, 1], stddev=0.1))
    b = tf.Variable(0.0)

    return tf.add(tf.matmul(input_tensor, W), b)

def build_model():
    n = 1 #Number of features
    X = tf.placeholder(tf.float32, [None, n])
    Y = tf.placeholder(tf.float32, [None, 1])

    layer = hidden_layer(X, 300)
    layer = hidden_layer(X, 200)
    layer = hidden_layer(layer, 20)
    Y_hat = readout_layer(layer)

    loss = tf.reduce_mean(tf.square(Y_hat - Y))
    model = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    return model, X, Y, Y_hat, loss

# Sample imput data. y = 3.x^2 + 4
#1000 samples from 0 to 100
train_x = np.random.rand(1000, 1) * 100
train_y = np.square(train_x) * 3.0 + 4.0

with tf.Session() as sess:
    model, X, Y, Y_hat, loss = build_model()
    sess.run(tf.global_variables_initializer())

    for train_step in range(40001):
        sess.run(model, {X:train_x, Y:train_y})

        # Print training progress
        if train_step % 2000 == 0:
            error_rate = sess.run(loss, {X:train_x, Y:train_y})
    
            print("Step:", train_step, 
                "Loss:", error_rate)
            if error_rate < 0.0001:
                break

    # Validate the model with data not used in training
    x_unseen = np.array([[6.0], [7.0], [8.0]])
    y_expected = np.square(x_unseen) * 3.0 + 4.0
    print("Predections:", sess.run(Y_hat, {X:x_unseen}))
    print("Expected:", y_expected)
