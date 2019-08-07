import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

price_data = pd.read_csv('train_price.csv')
price_data["price"].plot(kind = 'hist', grid = True)
plt.title("Log value of prices")
plt.show(block=False)

plt.figure()
actual_price = np.exp(price_data)
actual_price["price"].plot(kind = 'hist', grid = True)
plt.title("Actual prices")
plt.show()