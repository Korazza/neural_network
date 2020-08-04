from korynn import layers, activation
import numpy as np
import matplotlib.pyplot as plt


def create_data(points, classes):
    x = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, points)
            + np.random.randn(points) * 0.2
        )
    return x, y


def main():

    layer1 = layers.Dense(4, 5)
    layer2 = layers.Dense(5, 1)

    relu1 = activation.Relu()
    relu2 = activation.Relu()

    layer1.forward(X)
    relu1.forward(layer1.output)
    layer2.forward(relu1.output)
    relu2.forward(layer2.output)

    print(layer2.output)


main()
