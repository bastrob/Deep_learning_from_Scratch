import numpy as np
from numpy import ndarray

class Loss(object):
    """
    The "loss" of a neural network
    """

    def __init__(self):
        """
        Pass
        """
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        """
        Computes the actual loss value
        :param prediction:
        :param target:
        :return:
        """

        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self) -> ndarray:
        """
        Computes gradient of the loss value with respect to the input to the loss function
        :return:
        """
        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:
        """
        Every subclass of "Loss" must implement the _output function
        :return:
        """
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        """
        Every subclass of "Loss" must implement the _input_grad function
        :return:
        """
        raise NotImplementedError()

def assert_same_shape(array: ndarray,
                      array_grad: ndarray):
    assert array.shape == array_grad.shape, \
        '''
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        '''.format(tuple(array_grad.shape), tuple(array.shape))
    return None