import tensorflow.compat.v1 as tf
import pickle

def build_model():
    # Define the numeric feature columns
    numeric_features = [
        'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms',
        'beds', 'security_deposit', 'cleaning_fee', 'minimum_nights',
        'number_of_reviews', 'review_scores_value'
    ]

    numeric_columns = [
        tf.feature_column.numeric_column(key=feature)
        for feature in numeric_features
    ]

    # Define the category feature columns
    categorical_features = [
        'host_is_superhost', 'neighbourhood_cleansed', 'property_type',
        'room_type', 'bed_type', 'instant_bookable'
    ]

    # Load vocabulary of category features
    with open("vocabulary.pkl", "rb") as f:
        vocabulary = pickle.load(f)

    categorical_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocabulary[feature])
        for feature in categorical_features
    ]

    all_columns = numeric_columns + categorical_columns

    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=all_columns, model_dir="linear_regressor")

    return linear_regressor
