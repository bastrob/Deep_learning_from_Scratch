import numpy as np
from numpy import ndarray

from NN.ParamOperation import ParamOperation


class WeightMultiply(ParamOperation):
    """
    Weight multiplication operation for a neural network
    """

    def __init__(self, W: ndarray):
        """
        Initialize Operation with  self.param = W
        :param W:
        """
        super().__init__(W)

    def _output(self) -> ndarray:
        """
        Compute output.
        :return:
        """
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute input gradient.
        :param output_grad:
        :return:
        """
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute parameter gradient
        :param output_grad:
        :return:
        """
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)

    