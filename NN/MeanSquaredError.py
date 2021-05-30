import numpy as np
from numpy import ndarray

from NN.Loss import Loss


class MeanSquaredError(Loss):

    def __init__(self):
        """
        Pass
        """
        pass

    def _output(self) -> float:
        """
        Computes the per-observation squared error loss
        :return:
        """
        loss = np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

        return loss

    def _input_grad(self) -> ndarray:
        """
        Computes the loss gradient with respect to the input for MSE loss
        :return:
        """

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]
