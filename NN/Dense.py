from NN.BiasAdd import BiasAdd
from NN.Layer import Layer
from NN.Operation import Operation
from NN.Sigmoid import Sigmoid
import numpy as np

from WeightMultiply import WeightMultiply


class Dense(Layer):
    """
    A fully connected layer that inherits from "Layer"
    """

    def __init__(self, neurons: int, activation: Operation = Sigmoid()) -> None:
        """
        Requires an activation function upon initialization
        :param neurons:
        :param activation:
        """
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: int) -> None:
        """
        Defines the operations of a fully connected layer
        :param num_in:
        :return:
        """
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        return None
