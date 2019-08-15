import numpy as np
import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

graph = tf.add(a, b)

with tf.Session() as sess:
    result = sess.run(graph, {a:10.0, b:20.0})

print("Result:", result)

i = tf.Variable(0.0)
result = tf.add(i, 1)
graph = tf.assign(i, result)

with tf.Session() as sess:
    #initialize all variables
    sess.run(tf.global_variables_initializer())

    for step in range(5):
        sess.run(graph)

        print("Step:", step, "i:", sess.run(i))

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [2, None])
graph = tf.matmul(X, Y)
 
x_in = [
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
]
y_in = [
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]

with tf.Session() as sess:
    result = sess.run(graph, {X: x_in, Y: y_in})

print(result)