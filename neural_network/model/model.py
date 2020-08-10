from neural_network.layers import Layer
from neural_network.losses import Loss
from matplotlib import pyplot as plt
from typing import Union
import numpy as np


id_counter = 1


class Model:
    def __init__(self, loss: Union[str, Loss] = "mean_squared_error"):
        global id_counter
        self.id = self.__class__.__name__.lower() + "_" + str(id_counter)
        id_counter += 1
        self.layers = []
        self.loss = (
            getattr(
                getattr(__import__("neural_network"), "losses"),
                loss.replace("_", " ").title().replace(" ", ""),
            )()
            if isinstance(loss, str)
            else loss
        )

    def add(self, layer: Layer):
        if not layer.input_shape:
            if len(self.layers) > 0:
                layer.set_input_shape(self.layers[-1].units)
            else:
                raise Exception("First layer has no input_shape")
        self.layers.append(layer)

    def summary(self):
        print("\n ********* Model {} *********".format(self.id))
        print("\nLayers:", len(self.layers))
        for layer in self.layers:
            layer.summary()

    def predict(self, x: list, verbose: int = 0) -> np.ndarray:
        z = np.array(x).copy()
        for layer in self.layers:
            z = layer.forward(z)
            if verbose > 0:
                print("\nLayer {} output:\n".format(layer.id), z)
        return z

    def backpropagate(self, learning_rate: float, dcost_dpred: np.ndarray, x: np.ndarray):
        for i in range(len(self.layers)):
            layer = self.layers[-i - 1]
            prev_layer = self.layers[-i - 2] if i < len(self.layers) - 1 else None
            dpred_dz = layer.activation.d(layer.z)
            dz_dw = prev_layer.output.T if prev_layer else x.T
            dz_db = 1
            dcost_dw = dcost_dpred * dpred_dz * dz_dw
            dcost_db = dcost_dpred * dpred_dz * dz_db
            new_weights = []
            new_biases = []
            for w in range(len(layer.weights)):
                new_weights.append(layer.weights[w] - learning_rate * dcost_dw[w])
            for b in range(len(layer.biases)):
                new_biases.append(layer.biases[b] - learning_rate * dcost_db[b])
            layer.weights = np.array(new_weights)
            layer.biases = np.array(new_biases)

    def train(self, xs: list, ys: list, learning_rate: float = 0.01, epochs: int = 1, verbose: int = 0):
        xs = np.array(xs).astype(float)
        ys = np.array(ys).astype(float)
        costs = []
        for epoch in range(epochs):
            index = np.random.choice(xs.shape[0], size=1, replace=False)
            x = xs[index]
            y = ys[index]
            prediction = self.predict(x)
            cost = self.loss.f(y, prediction)
            costs.append(cost)
            dcost_dpred = self.loss.d(y, prediction)
            self.backpropagate(learning_rate, dcost_dpred, x)
            if verbose >= 2:
                print("\nEpoch", epoch)
                print("\nCost:", cost)
        if verbose >= 1:
            print("\nLast Cost:", costs[-1])
            plt.plot(costs)
            plt.xlabel("Epochs")
            plt.ylabel("Cost")
            plt.show()
