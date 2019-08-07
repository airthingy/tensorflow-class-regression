import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import pickle

def build_model():
    numeric_columns = [
        'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms',
        'beds', 'security_deposit', 'cleaning_fee', 'minimum_nights',
        'number_of_reviews', 'review_scores_value'
    ]
    categorical_columns = [
        'host_is_superhost', 'neighbourhood_cleansed', 'property_type',
        'room_type', 'bed_type', 'instant_bookable'
    ]
    
    with open("vocabulary.pkl", "rb") as f:
        vocabulary = pickle.load(f)

    print(vocabulary)

build_model()