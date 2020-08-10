from neural_network import layers, model
from matplotlib import pyplot as plt

# Testing my neural_network module with XOR problem

# inputs
x = [[0, 0], [0, 1], [1, 0], [1, 1]]

# labels
y = [0, 1, 1, 0]

model = model.Model(loss="binary_cross_entropy")
model.add(layers.Dense(2, input_shape=2, activation="sigmoid"))
model.add(layers.Dense(2, activation="sigmoid"))
model.add(layers.Dense(1, activation="sigmoid"))

model.train(x, y, learning_rate=0.01, epochs=100000, verbose=2)

for x_test in x:
    print(x_test, model.predict(x_test))
