import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

model = keras.Sequential()
model.add(layers.Dense(50, input_dim=1, activation='relu'))
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dense(1)) #Readout

model.compile(loss='mean_squared_error',
                optimizer="adam")

train_x = np.random.rand(100, 1) * 5
train_y = np.square(train_x) * 3.0 + 4.0

model.fit(train_x, train_y, epochs=5000)

x_unseen = np.array([[6.0], [7.0], [8.0]])
y_expected = np.square(x_unseen) * 3.0 + 4.0

test_predictions = model.predict(x_unseen)

print("Predections:", test_predictions)
print("Expected:", y_expected)