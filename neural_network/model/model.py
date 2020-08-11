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
        if not layer.input_dim:
            if len(self.layers) > 0:
                layer.set_input_dim(self.layers[-1].units)
            else:
                raise Exception("First layer has no input_dim")
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
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1] if i > 0 else None
            dpred_dz = layer.activation.derivative(layer.z)
            dz_dw = prev_layer.output.T if prev_layer else x.T
            dz_db = 1
            dcost_dw = dcost_dpred * dpred_dz * dz_dw
            dcost_db = dcost_dpred * dpred_dz * dz_db
            new_weights = []
            new_biases = []
            layer.weights -= learning_rate * dcost_dw
            layer.biases -= learning_rate * dcost_db

    def train(self, xs: list, ys: list, learning_rate: float = 0.01, epochs: int = 1, verbose: int = 0):
        xs = np.array(xs).astype(float)
        ys = np.array(ys).astype(float)
        costs = []
        for epoch in range(epochs):
            index = np.random.choice(xs.shape[0], size=1, replace=False)
            x = xs[index]
            y = ys[index]
            prediction = self.predict(x)
            cost = self.loss(y, prediction)
            costs.append(cost)
            dcost_dpred = self.loss.derivative(y, prediction)
            self.backpropagate(learning_rate, dcost_dpred, x)
            if verbose >= 2:
                print("Epoch {} | Cost: {}".format(epoch + 1, cost), end="\r")
        if verbose >= 1:
            print("Training results | Average Cost: {}\n".format(np.mean(costs)))
            plt.plot(costs)
            plt.title("Cost over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Cost")
            plt.show()

    def plot_decision_boundary(self, x, y, steps=1000, cmap='RdYlBu'):
        cmap = plt.get_cmap(cmap)
        x = np.array(x)
        y = np.array(y)
        xmin, xmax = x[:, 0].min() - 1, x[:, 0].max() + 1
        ymin, ymax = x[:, 1].min() - 1, x[:, 1].max() + 1
        x_span = np.linspace(xmin, xmax, steps)
        y_span = np.linspace(ymin, ymax, steps)
        xx, yy = np.meshgrid(x_span, y_span)
        labels = self.predict(np.c_[xx.ravel(), yy.ravel()])
        z = labels.reshape(xx.shape)
        fig, ax = plt.subplots()
        ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)
        train_labels = self.predict(x)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap, lw=0)
        plt.title("Decision boundary output")
        plt.show()
