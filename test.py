from neural_network import layers, model, losses
import numpy as np
from matplotlib import pyplot as plt

# Testing my neural_network module with XOR problem

# inputs
x = [[0, 0], [0, 1], [1, 0], [1, 1]]

# labels
y = [0, 1, 1, 0]

model = model.Model(loss="mean_squared_error")
model.add(layers.Dense(2, input_shape=2, activation="sigmoid"))
model.add(layers.Dense(2, activation="sigmoid"))
model.add(layers.Dense(1, activation="sigmoid"))

model.train(x, y, learning_rate=0.02, epochs=10000, verbose=1)

for x_test in x:
    print(x_test, model.predict(x_test))
