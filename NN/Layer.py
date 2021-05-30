from numpy import ndarray
from typing import List

from NN.Operation import Operation
from NN.ParamOperation import ParamOperation


class Layer(object):
    """
    A "layer" of neuros in a neural network
    """

    def __init__(self, neurons: int):
        """
        The number of "neurons" roughly corresponds to the "breadth" of the layer
        :param neurons:
        """
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.params_grad: List[ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, num_in: int) -> None:
        """
        The setup_layer function must be implemented for each layer
        :param num_in:
        :return:
        """
        raise NotImplementedError()

    def forward(self, input_: ndarray) -> ndarray:
        """
        Passes input forward through a series of operations.
        :param input_:
        :return:
        """
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Passes output_grad backward through a series of operations
        Check appropriate shapes
        :param output_grad:
        :return:
        """

        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self) -> ndarray:
        """
        Extracts the _param_grads from a layer's operations
        :return:
        """

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> ndarray:
        """
        Extract the _params from a layer's operations
        :return:
        """
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)

def assert_same_shape(array: ndarray,
                      array_grad: ndarray):
    assert array.shape == array_grad.shape, \
        '''
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        '''.format(tuple(array_grad.shape), tuple(array.shape))
    return None