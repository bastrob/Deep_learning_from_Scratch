import numpy as np
from numpy import ndarray

from NN.ParamOperation import ParamOperation


class BiasAdd(ParamOperation):
    """
    Compute bias addition
    """

    def __init__(self, B: ndarray):
        """
        Initialize Operation with a self.param = B
        Check appropriate shape
        :param B:
        """

        assert B.shape[0] == 1

        super().__init__(B)

    def _output(self) -> ndarray:
        """
        Compute output
        :return:
        """
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute input gradient
        :param output_grad:
        :return:
        """
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute parameter gradient
        :param output_grad:
        :return:
        """
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])