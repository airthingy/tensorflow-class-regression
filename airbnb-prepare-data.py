import pandas as pd
import numpy as np

used_features = [
    'property_type', 'room_type', 'bathrooms', 'bedrooms', 'beds', 'bed_type',
    'accommodates', 'host_total_listings_count', 'number_of_reviews',
    'review_scores_value', 'neighbourhood_cleansed', 'cleaning_fee',
    'minimum_nights', 'security_deposit', 'host_is_superhost',
    'instant_bookable', 'price'
]

source_data = pd.read_csv('listings.csv', usecols=used_features)

# Convert string to numerical data
for feature in ["cleaning_fee", "security_deposit", "price"]:
    source_data[feature] = source_data[feature].map(
        lambda x: x.replace("$", '').replace(",", ''), na_action='ignore')
    source_data[feature] = source_data[feature].astype(float)
    source_data[feature].fillna(source_data[feature].median(), inplace=True)

for feature in ["bathrooms", "bedrooms", "beds", "review_scores_value"]:
    source_data[feature].fillna(source_data[feature].median(), inplace=True)

source_data['property_type'].fillna('Apartment',inplace = True)

#Filter out too expensive and too cheap properties
source_data = source_data[(source_data["price"]>50)&(source_data["price"]<500)]

# Split source into training and test data
train = source_data.sample(frac=0.8,random_state=200)
test = source_data.drop(train.index)

train_price = train["price"]
test_price = test["price"]
train_features = train.drop("price", axis=1)
test_features = test.drop("price", axis=1)

# Spread out price by taking a log. Otherwise prices are very close to each other
# which makes prediction less accurate
train_price = np.log(train_price)
test_price = np.log(test_price)

# Save split data in files
train_price.to_csv("train_price.csv", header=True, index=False)
test_price.to_csv("test_price.csv", header=True, index=False)
train_features.to_csv("train_features.csv", header=True, index=False)
test_features.to_csv("test_features.csv", header=True, index=False)