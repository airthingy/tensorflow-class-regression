import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import pickle


def build_model():
    # Define the numeric feature columns
    numeric_columns = [
        'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms',
        'beds', 'security_deposit', 'cleaning_fee', 'minimum_nights',
        'number_of_reviews', 'review_scores_value'
    ]

    numeric_features = [
        tf.feature_column.numeric_column(key=column)
        for column in numeric_columns
    ]

    # Define the category feature columns
    categorical_columns = [
        'host_is_superhost', 'neighbourhood_cleansed', 'property_type',
        'room_type', 'bed_type', 'instant_bookable'
    ]

    # Load vocabulary of category features
    with open("vocabulary.pkl", "rb") as f:
        vocabulary = pickle.load(f)

    categorical_features = [
        tf.feature_column.categorical_column_with_vocabulary_list(
            key=column, vocabulary_list=vocabulary[column])
        for column in categorical_columns
    ]

    linear_features = numeric_features + categorical_features

    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=linear_features, model_dir="linear_regressor")

    return linear_regressor


def train_model():
    linear_regressor = build_model()
    train_features = pd.read_csv("train_features.csv")
    train_prices = pd.read_csv("train_price.csv")

    training_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_features,
                                                            y=train_prices["price"],
                                                            batch_size=32,
                                                            shuffle=True,
                                                            num_epochs=None)
    linear_regressor.train(input_fn = training_input_fn,steps=2000)

def evaluate_model():
    linear_regressor = build_model()
    test_features = pd.read_csv("test_features.csv")
    test_prices = pd.read_csv("test_price.csv")

    eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_features,
                                                            y=test_prices["price"],
                                                            batch_size=32,
                                                            shuffle=False,
                                                            num_epochs=1)
    result = linear_regressor.evaluate(input_fn = eval_input_fn)
    print(result)

def predict():
    linear_regressor = build_model()
    test_features = pd.read_csv("test_features.csv")

    predict_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_features,
                                                            y=None,
                                                            batch_size=32,
                                                            shuffle=False,
                                                            num_epochs=1)
    results = linear_regressor.predict(input_fn = predict_input_fn)
    
    for r in results:
        print(r["predictions"])

predict()