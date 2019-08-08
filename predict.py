import pandas as pd
import tensorflow.compat.v1 as tf
import model

def predict():
    linear_regressor = model.build_model()
    test_features = pd.read_csv("test_features.csv")

    predict_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_features,
                                                            y=None,
                                                            batch_size=32,
                                                            shuffle=False,
                                                            num_epochs=1)
    results = linear_regressor.predict(input_fn = predict_input_fn)
    
    for r in results:
        print(r["predictions"][0])

predict()