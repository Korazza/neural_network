from neural_network import layers, model
from matplotlib import pyplot as plt
import numpy as np

# Testing my neural_network module with XOR problem

# inputs
x = [[0, 0], [0, 1], [1, 0], [1, 1]]

# labels
y = [0, 1, 1, 0]

model = model.Model(loss="mean_squared_error")
model.add(layers.Dense(3, input_dim=2, activation="sigmoid"))
model.add(layers.Dense(3, activation="sigmoid"))
model.add(layers.Dense(1, activation="sigmoid"))

model.train(x, y, learning_rate=0.01, epochs=80000, verbose=2)

model.plot_decision_boundary(x, y)
