class Optimizer(object):
    """
    Base class for a neural network optimizer
    """

    def __init__(self, lr: float = 0.01):
        """
        Every optimizer must have an initial learning rate
        :param lr:
        """
        self.lr = lr

    def step(self) -> None:
        """
        Every optimizer must implement the "step" function
        :return:
        """
        pass
