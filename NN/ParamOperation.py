from numpy import ndarray
from NN.Operation import Operation


class ParamOperation(Operation):
    """
    An Operation with parameters
    """

    def __init__(self, param: ndarray) -> ndarray:
        """
        The ParamOperation method
        :param param:
        """
        super().__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Calls the self._input_grad() function.
        Checks that the appropriate shapes match
        :param output_grad:
        :return:
        """
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Every subclass of ParamOperation must implement _param_grad
        :param output_grad:
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
