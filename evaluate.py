import pandas as pd
import tensorflow.compat.v1 as tf
import model

def evaluate_model():
    linear_regressor = model.build_model()
    test_features = pd.read_csv("test_features.csv")
    test_prices = pd.read_csv("test_price.csv")

    eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_features,
                                                            y=test_prices["price"],
                                                            batch_size=32,
                                                            shuffle=False,
                                                            num_epochs=1)
    result = linear_regressor.evaluate(input_fn = eval_input_fn)
    print(result)

evaluate_model()