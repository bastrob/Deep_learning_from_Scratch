from NN.Optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer
    """

    def __init__(self, lr: float = 0.01) -> None:
        """
        Pass
        :param lr:
        """
        super().__init__(lr)

    def step(self):
        """
        For each parameter, adjust in the appropriate direction,
        with the magnitude of the adjustement based on the learning rate
        :return:
        """
        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):
            param -= self.lr * param_grad
