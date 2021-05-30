import numpy as np
from numpy import ndarray

from NN.Dense import Dense
from NN.Linear import Linear
from NN.MeanSquaredError import MeanSquaredError
from NN.NeuralNetwork import NeuralNetwork
from NN.SGD import SGD
from NN.Sigmoid import Sigmoid

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from NN.Trainer import Trainer


def mae(y_true: ndarray, y_pred: ndarray):
    """
    Compute mean absolute error for a neural network.
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: ndarray, y_pred: ndarray):
    """
    Compute root mean squared error for a neural network.
    """
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def eval_regression_model(model: NeuralNetwork,
                          X_test: ndarray,
                          y_test: ndarray):
    """
    Compute mae and rmse for a neural network.
    """
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(mae(preds, y_test)))
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))
    print()


def to_2d_np(a: ndarray,
             type: str = "col") -> ndarray:
    """
    Turns a 1D Tensor into 2D
    """

    assert a.ndim == 1, \
        "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)


boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names

# Scaling the data
s = StandardScaler()
data = s.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

# make target 2d array
y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)

lr = NeuralNetwork(
    layers=[Dense(neurons=1,
                  activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

nn = NeuralNetwork(
    layers=[Dense(neurons=13,
                  activation=Sigmoid()),
            Dense(neurons=1,
                  activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

dl = NeuralNetwork(
    layers=[Dense(neurons=13,
                  activation=Sigmoid()),
            Dense(neurons=13,
                  activation=Sigmoid()),
            Dense(neurons=1,
                  activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

trainer = Trainer(lr, SGD(lr=0.01))

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(lr, X_test, y_test)

trainer = Trainer(nn, SGD(lr=0.01))

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(nn, X_test, y_test)

trainer = Trainer(dl, SGD(lr=0.01))

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(dl, X_test, y_test)
