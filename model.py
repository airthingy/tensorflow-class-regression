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
