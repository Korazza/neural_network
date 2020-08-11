from neural_network import layers, model

# Testing my neural_network module with XOR problem

# inputs
x = [[0, 0], [0, 1], [1, 0], [1, 1]]

# labels
y = [0, 1, 1, 0]

model = model.Model(loss="binary_cross_entropy")
model.add(layers.Dense(3, input_dim=2, activation="sigmoid"))
model.add(layers.Dense(3, activation="sigmoid"))
model.add(layers.Dense(1, activation="sigmoid"))

model.train(x, y, learning_rate=0.013, epochs=100000, verbose=2)

model.plot_decision_boundary(x, y)
