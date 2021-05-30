from numpy import ndarray
from NN.Operation import Operation


class Linear(Operation):
    """
    "Identity" activation function
    """

    def __init__(self) -> None:
        """Pass"""
        super().__init__()

    def _output(self) -> ndarray:
        """Pass through"""
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """Pass through"""
        return output_grad