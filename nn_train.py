import pandas as pd
import tensorflow.compat.v1 as tf
import nn_model

def train_model():
    nn_regressor = nn_model.build_model()
    train_features = pd.read_csv("train_features.csv")
    train_prices = pd.read_csv("train_price.csv")

    training_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_features,
                                                            y=train_prices["price"],
                                                            batch_size=32,
                                                            shuffle=True,
                                                            num_epochs=None)
    nn_regressor.train(input_fn = training_input_fn,steps=2000)

train_model()