This project shows how to do linear regression using Tensorflow.

# Getting Started
## Installation
Install Python3 and pip3.

Run this command to verify Python3 is installed.

```
python3 --version
```

Install these packages:

```
pip3 install tensorflow
pip3 install pandas
pip3 install matplotlib
```

## Choosing an Editor
It is highly recommended that you use a modern editor like Visual Studio Code, Sublime Text and PyCharm.

Install Python development plugins for these editors.

## How to do the Workshop?
As you can already see completed solution code is already given to you in this repo. Use them only as a guide in case your own code is not working for some reason. You can also copy paste lengthy tedius code that serves no educational purpose.

Create a folder called **workshop** and do all your work there.

> **Tip:** Make sure you understand every line of code that goes into your work. Blindy copying and pasting code from the completed solution won't help you in any way.


# Workshop - Tensorflow Basics
In this workshop we will learn these basic concepts of Tensorflow:

- **Placeholder** - This is where you supply values as input to a computation. For example when training a model you supply training data from files into placeholders.
- **Variable** - Tensorflow computes data and saves them in variables. For example if a model predicts the chance of rain that will be saved in a variable. Think of variables as computed outputs.
- **Computation graph** - All algebric operations are declared in a graph like data structure made up of operands and operators. Once a graph is created it can be executed anytime and any number of times. This is different from the way conventional programming languages like Java and C++ perform computation. There are tremendous benefits to graph based computation:
    - Some operations like matrix multiplication can be distributed to GPUs for massively parallel computation. You as a developer don't have to know anything about parallel computing or GPU programming.
    - Operations can be federated, meaning, they can be distributed across multiple machines over the network.
- **Matrix operations** - Linear algebra is at the heart of machine learning. Before we get into full scale ML problems we will learn how Tensorflow does basic matrix operations.

## Placeholder and Graph
Let's do one of the most basic algebric operations - adding two numbers.

Create a file called ``basic.py``.

Add these lines to import necessary package names.

```python
import numpy as np
import tensorflow as tf
```

Create two placeholders and declare a computation graph that adds these two numbers.

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

graph = tf.add(a, b)
```

Now we are ready to execute the graph. Execution requires a Tensorflow session. Add these lines.

```python
with tf.Session() as sess:
    result = sess.run(graph, {a:10.0, b:20.0})

print("Result:", result)
```

Note how we are supplying the values for the placeholders using a dictionary. The keys in this dictionary must match the placeholder names.

Save the file.

Run the code like this.

```
python3 basic.py
```

You will see many deprecation warnings. You can ignore them. But make sure you see the result printed out.

```
Result: 30.0
```

Congratulations! You have successfully solved your first problem in Tensorflow. This may not seem like much. But when these operations are carried out in parallel in a gigantic scale you start to appreciate Tensorflow.

> **Did you know?** All participants in a graph, like placeholders, variables, and the graph itself is a ``tf.Tensor``. The result of running a graph may or may not be a tensor. In our example above the result is a simple ``float32``. 

## Variable
Variables are where results of computation can be saved. Let's model this operation that increments a variable.

```python
i = i + 1
```

You may recall from your CompSci 101 days that this is actually a combination of two operations:

- Add 1 to ``i`` and save the result in a temporary variable (created behind the scene by the compiler).
- Assign the result to the variable ``i``.

Let's create a graph that models this problem. At the bottom of ``basic.py`` add these lines.

```python
i = tf.Variable(0.0)
result = tf.add(i, 1)

graph = tf.assign(i, result)
```

Now we can execute the graph. Add these lines.

```python
with tf.Session() as sess:
    #initialize all variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(5):
        sess.run(graph)

        print("Step:", step, "i:", sess.run(i))
```

A few things to note here.

- Even though we declared ``0.0`` as the initial value of ``i``, we still had to execute ``tf.global_variables_initializer()`` to initialize the variables. It is at this point ``0.0`` gets assigned to ``i``.
- To read the value of a variable you need to execute the variable as a graph. We are doing that here using ``sess.run(i)``.

Save and run the file. You should see this.

```
Step: 0 i: 1.0
Step: 1 i: 2.0
Step: 2 i: 3.0
Step: 3 i: 4.0
Step: 4 i: 5.0
```

>**Advanced:** You can think of variables as where we can store state in a lengthy series of operations. During training weights and biases are constantly updated for the most optimal outcome. As you can imagine weights and biases are always defined as variables in Tensorflow. These variables are computed only during the training phase. Their final values are saved in files at the end of training. During prediction phase their values are restored from files and assigned to the variables. This is why training takes so long - hours to weeks depending on the complexity of the model. Prediction can be done in the blink of an eye.
>
>Some advanced neural networks like RNN and LSTM need to maintain state even during the prediction phase. For example a model that translates language the meaning of a sentence can depend on the meaning of previous few sentences.

## Matrix Operation
Let's multiply a 4x2 matrix with a 2x4 matrix. The result should be a 4x4 matrix.

At the end of ``basic.py`` define the graph.

```python
X = tf.placeholder(tf.float32, [4, 2])
Y = tf.placeholder(tf.float32, [2, 4])

graph = tf.matmul(X, Y)
```

Note, this time we specified the dimensions (also called shape) of the input data.

Enter the input data.

```python
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
```

Finally execute the graph and display the result.

```python
with tf.Session() as sess:
    result = sess.run(graph, {X: x_in, Y: y_in})

print(result)
```

Save and run the file. Make sure you see this result.

```
[[11. 14. 17. 20.]
 [23. 30. 37. 44.]
 [35. 46. 57. 68.]
 [47. 62. 77. 92.]]
```

>**Numpy Array:** Here we are feeding plain Python lists to the placeholders. Tensorflow also allows feeding numpy arrays. In real life you will mostly work with numpy arrays.

Note: 

- As of Python 3.5 you can use the ``@`` operator for matrix multiplication. Example: ``X @ Y``. But I recommend you keep using ``tf.matmul()`` for better readability.
- To do elementwise multiplication of two matrices of same dimension you use ``tf.multiply(X, Y)`` or just the ``X * Y`` syntax.

### Unknown Dimensions
In the code above we have precisely stated the matrix dimensions as 4x2 and 2x4. In real life there will be some dimensions that you will not know when writing the code. For example, you may not know how many sample data are available in the training dataset. Tensorflow is very flexible in this regard. You can use ``None`` as the dimension in those cases. At execution time Tensorflow will make sure that the actual dimensions of the data fed to the model are valid for the requested matrix operation.

Change the way ``X`` and ``Y`` placeholders are defined.

```python
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [2, None])
```

Now at execution time the missing dimensions will be filled in based on the data that is fed to the model.

Save and run the file. You should see the same result.

# Workshop - Simple Linear Regression
In linear regression system learns weights and bias from training data such that it can fit a line through the data most accurately (with least amount of error). Using this technique you can solve problems like housing price prediction.

During training error (also called loss) is gradually minimized using a technique called Gradient Descent.

Linear regression may be one of the simplest ML techniques but it forms the foundation for other learning algorithms. This is why we need to spend a bit of time fully understanding how this works.

In this workshop we will solve a very simple problem. We will train the system using test data that we know follows this equation.

```
y = 3x + 4
```

At the end of training the model should discover that the weight is 3.0 and bias is 4.0.

We keep the problem purposely simple. The focus of this workshop is:

- How to run training that discovers the weights and biases for least error.
- How to run prediction
- How to save and restore weights and biases

## Define the Model
Create a file called ``simple-regression.py``.

Add these import statements.

```python
import tensorflow.compat.v1 as tf
import numpy as np
```

In our problem we have only one weight and one bias. Let's declare them as a 1x1 matrix.

```python
# Weight and bias variables as 1x1 matrix initialized to 0
W = tf.Variable([[0.0]])
b = tf.Variable([[0.0]])
```

>**Important:** ``tf.matmul()`` expects the matrices to be at minimum 2D. This is why we had to create ``W`` and ``b`` as 1x1 matrix and not a one dimensional vector.

Next, declare the input placeholders. They have dimension of mx1 where m is the number of training samples (not known when writing code).

```python
X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])
```

Finally, declare the model like this.

```python
predictions = tf.add(tf.matmul(X, W), b)
loss = tf.reduce_mean(tf.square(predictions - Y))
model = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
```

## Generate Training Data
Normally training data is loaded from disk. But in our simple example we can just generate it. Add these lines.

```python
# Sample imput data. y = 3x + 4
train_x = np.array([[1.0], [2.0], [3.0], [4.0]])
train_y = train_x * 3.0 + 4.0
```

We will feed ``train_x`` to ``X`` and ``train_y`` to ``Y``. Verify that their dimensions match up.

## Training and Prediction
Add this code.

```python
with tf.Session() as sess:
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
                break

    # Validate the model with data not used in training
    x_unseen = np.array([[6.0], [7.0], [8.0]])
    y_expected = x_unseen * 3.0 + 4.0
    print("Predections:", sess.run(predictions, {X:x_unseen}))
    print("Expected:", y_expected)
```

## Run Code
Save your file and run it like this.

```
python3 simple-regression.py
```

Note as training progresses the weight and bias converge on 3 and 4 respectively. Also verify that prediction on unseen data is very close to what is expected.

## Separate Training and Prediction Phases
Right now our code is running training and prediction in the same Tensorflow session. That is not how things work in real life. Training can take hours to days depending on how complex the model is and how much training data you have. Prediction is actually used by the end user perhaps from a web site or a microservice. Code for these two phases are developed and deployed independently. 

Basic steps to separate these two phases go like this:

- At the end of training save the weights and biases (collectively called parameters) to disk.
- During prediction these weights and biases are loaded from disk. Variables are initialized with these values. As a result, these parameter files need to be deployed to production along with the prediction code.

Below the line:

```python
with tf.Session() as sess:
```

Add:

```python
saver = tf.train.Saver()
```

Save the parameters when training is done.

```python
if error_rate < 0.0001:
    saver.save(sess, "./model.ckpt")

    break
```

Move the prediction code in its own separate session scope.

```python
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
```

Save file and run it. The result will be the same as before.

>**Tip:** The ``predictions`` node of the graph is used during the prediction phase. The ``loss`` and ``model`` nodes are not useful during prediction. In any case, much of the graph definition code is shared by the training and prediction phases. As a result you need to find a way to isolate this code in a reusable file. It is also possible that the prediction phase is coded using a different programming language. For example, if the end user web site is created using Java you will need to re-write the graph creation code using Java.

## Training Data
This repo contains AirBnb data for Boston in ``listings.csv``. To update this data or
work with another city go to: http://insideairbnb.com/get-the-data.html.