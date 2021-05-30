from typing import List
import numpy as np
from numpy import ndarray

from NN.Dense import Dense
from NN.Layer import Layer
from NN.Linear import Linear
from NN.Loss import Loss
from NN.MeanSquaredError import MeanSquaredError
from NN.Sigmoid import Sigmoid


class NeuralNetwork(object):
    """
    The class for a neural network
    """
    def __init__(self, layers: List[Layer],
                 loss: Loss,
                 seed: float = 1):
        """
        Neural networks need layers, and a loss.
        :param layers:
        :param loss:
        :param seed:
        """
        self.layers = layers
        self.loss = loss
        self.seed = seed

        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: ndarray) -> ndarray:
        """
        Passes data forward through a series of layers
        :param x_batch:
        :return:
        """
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_grad: ndarray) -> None:
        """
        Passes data backward through a series of layers
        :param loss_grad:
        :return:
        """
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    def train_batch(self,
                    x_batch: ndarray,
                    y_batch: ndarray) -> float:
        """
        Passes data forward through the layers.
        Computes the loss
        :param x_batch:
        :param y_batch:
        :return:
        """
        predictions = self.forward(x_batch)

        loss = self.loss.forward(predictions, y_batch)

        self.backward(self.loss.backward())

        return loss

    def params(self):
        """
        Gets the parameters for the network
        :return:
        """
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        """
        Gets the gradient of the loss with respect to the parameters for the network.
        :return:
        """
        for layer in self.layers:
            yield from layer.param_grads