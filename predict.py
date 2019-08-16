import pandas as pd
import tensorflow.compat.v1 as tf
import numpy as np
import model

def predict():
    linear_regressor = model.build_model()
    test_features = pd.read_csv("test_features.csv")
    test_prices = pd.read_csv("test_price.csv")

    predict_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_features,
                                                            y=None,
                                                            batch_size=32,
                                                            shuffle=False,
                                                            num_epochs=1)
    results = linear_regressor.predict(input_fn = predict_input_fn)
    
    for pair in zip(results, test_prices.values):
        result = pair[0]
        predicted_price = result["predictions"][0]
        actual_price = pair[1][0]

        print("Predicted:", predicted_price, "Actual:", actual_price)

predict()