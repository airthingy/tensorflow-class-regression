import pandas as pd
import tensorflow.compat.v1 as tf
import logging
import model

def train_model():
    # Show training progress
    logging.getLogger().setLevel(logging.INFO)

    linear_regressor = model.build_model()
    train_features = pd.read_csv("train_features.csv")
    train_prices = pd.read_csv("train_price.csv")

    training_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_features,
                                                            y=train_prices["price"],
                                                            batch_size=128,
                                                            shuffle=True,
                                                            num_epochs=1000)
    linear_regressor.train(input_fn = training_input_fn)

train_model()