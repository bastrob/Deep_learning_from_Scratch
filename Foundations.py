import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from typing import Callable, List


def square(x: ndarray) -> ndarray:
    """
    Square each element in the input ndarray.
    :param x:
    :return:
    """
    return np.power(x, 2)


def leaky_relu(x: ndarray) -> ndarray:
    """
    Apply Leaky ReLU function to each element in ndarray
    :param x:
    :return:
    """
    return np.maximum(0.2 * x, x)


def sigmoid(x: ndarray) -> ndarray:
    """
    Apply the sigmoid function to each element in ndarray
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def deriv(func: Callable[[ndarray], ndarray], input_: ndarray, delta: float = 0.001) -> ndarray:
    """
    Evaluates the derivative of a function func at every element in the _input array.
    :param func:
    :param input_:
    :param delta:
    :return:
    """
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


# A function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[ndarray], ndarray]

# A chain is a list of functions
Chain = List[Array_Function]


def chain_length_2(chain: Chain, a: ndarray) -> ndarray:
    """
    Evaluates two functions in a row, in a Chain
    :param chain:
    :param a:
    :return:
    """
    assert len(chain) == 2, \
        "Length of input chain should be 2"

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(a))


def plot_chain(ax,
               chain: Chain,
               input_range: ndarray) -> None:
    '''
    Plots a chain function - a function made up of
    multiple consecutive ndarray -> ndarray mappings -
    Across the input_range

    ax: matplotlib Subplot for plotting
    '''

    assert input_range.ndim == 1, \
        "Function requires a 1 dimensional ndarray as input_range"

    output_range = chain_length_2(chain, input_range)
    ax.plot(input_range, output_range)


def plot_chain_deriv(ax, chain: Chain, input_range: ndarray) -> ndarray:
    '''
    Uses the chain rule to plot the derivative of a function consisting of two nested functions.

    ax: matplotlib Subplot for plotting
    '''
    output_range = chain_deriv_2(chain, input_range)
    ax.plot(input_range, output_range)


def chain_deriv_2(chain: Chain, input_range: ndarray) -> ndarray:
    """
    Uses the chain rule to compute the derivative of two nested functions:
    (f2(f1(x))' = f2'(f1(x)) * f1'(x)
    :param chain:
    :param input_range:
    :return:
    """
    assert len(chain) == 2, \
        "This function requires Chain objects of length 2"

    assert input_range.ndim == 1, \
        "Function requires a 1 dimensional ndarray as input_range"

    f1 = chain[0]
    f2 = chain[1]

    # df1/dx
    f1_of_x = f1(input_range)

    # df1/du
    df1dx = deriv(f1, input_range)

    # df2/du(f1(x))
    df2dx = deriv(f2, f1(input_range))

    # Multiplying these quantities together at each point
    return df1dx * df2dx


PLOT_RANGE = np.arange(-3, 3, 0.01)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 8))  # 2 Rows, 1 Col

chain_1 = [square, sigmoid]
chain_2 = [sigmoid, square]


# plot_chain(ax[0], chain_1, PLOT_RANGE)
# plot_chain_deriv(ax[0], chain_1, PLOT_RANGE)

# plot_chain(ax[1], chain_2, PLOT_RANGE)
# plot_chain_deriv(ax[1], chain_2, PLOT_RANGE)

# plt.show()

def chain_deriv_3(chain: Chain, input_range: ndarray) -> ndarray:
    """
    Uses the chain rule to compute the derivative of two nested functions:
    (f3(f2(f1(x)))' = f3'(f2(f1(x)) * f2'(f1(x)) * f1'(x)
    :param chain:
    :param input_range:
    :return:
    """
    assert len(chain) == 3, \
        "This function requires Chain objects of length 2"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    f1_of_x = f1(input_range)
    f2_of_x = f2(f1_of_x)

    df1dx = deriv(f1, input_range)

    df2du = deriv(f2, f1_of_x)

    df3du = deriv(f3, f2_of_x)

    return df1dx * df2du * df3du


arr = np.array([1, 2, 3, 4, 5])
chain_3 = [sigmoid, square, leaky_relu]

print(chain_deriv_3(chain_3, arr))


def multiple_inputs_add(x: ndarray, y: ndarray, sigma: Array_Function) -> float:
    """
    Function with multiple inputs and addition, forward pass.
    :param x:
    :param y:
    :param sigma:
    :return:
    """

    assert x.shape == y.shape

    a = x + y

    return sigma(a)


def multiple_inputs_add_backward(x: ndarray, y: ndarray, sigma: Array_Function) -> float:
    """
    Computes the derivative of this simple function with respect to both inputs.
    :param x:
    :param y:
    :param sigma:
    :return:
    """

    # Compute the forward pass
    a = x + y

    # Compute derivatives
    dsda = deriv(sigma, a)

    dadx, dady = 1, 1

    return dsda * dadx, dsda * dady


def matmul_forward(X: ndarray, W: ndarray) -> ndarray:
    """
    Computes the forward pass of a matrix multiplication
    :param X:
    :param W:
    :return:
    """
    assert X.shape[1] == W.shape[0], \
        """
        For matrix multiplication, the number of columns in the first array should
        match the number of rows in the second; instead the number of columns in the
        first array is {0} and the number of rows in the second array is {1}.
        """

    # matrix multiplication
    N = np.dot(X, W)

    return N


def matmut_backward_first(X: ndarray, W: ndarray) -> ndarray:
    """
    Computes the backward pass of a matrix multiplication with respect to the first argument.
    :param X:
    :param W:
    :return:
    """

    # backward pass
    dndX = np.transpose(W, (1, 0))

    return dndX


def matrix_forward_extra(X: ndarray, W: ndarray, sigma: Array_Function) -> ndarray:
    """
    Computes the forward pass of a function involving matrix multiplication, one extra function
    :param X:
    :param W:
    :param sigma:
    :return:
    """

    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma function
    S = sigmoid(N)

    return S


def matrix_function_backward_1(X: ndarray, W: ndarray, sigma: Array_Function) -> ndarray:
    """
    Computes the derivative of our matrix function with respect to the first element
    :param X:
    :param W:
    :param sigma:
    :return:
    """

    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # backward calculation
    dSdN = deriv(sigma, N)

    # dNdX
    dNdX = np.transpose(W, (1, 0))

    return np.dot(dSdN, dNdX)

np.random.seed(190203)

X = np.random.randn(1,3)
W = np.random.randn(3,1)

X = np.array([[ 0.47231121,  0.61514271, -1.72622715 + 0.01]])
print(X)
print(X.shape)
print(W.shape)


print(matrix_function_backward_1(X, W, sigmoid))
print(matrix_forward_extra(X, W, sigmoid))
print(-1.72622715)
print(-0.11206627)
print(-0.11206627 * 0.01)
print(0.89779986 - 0.11206627 * 0.01)

def matrix_function_forward_sum(X: ndarray, W: ndarray, sigma: Array_Function) -> float:
    """
    Computing the result of the forward pass of this function
    with input ndarrays X and W and function sigma
    :param X:
    :param W:
    :param sigma:
    :return:
    """

    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # sum all the elements
    L = np.sum(S)

    return L

def matrix_function_backward_sum_1(X: ndarray, W: ndarray, sigma: Array_Function) -> ndarray:
    """
    Compute derivative of matrix function with a sum with respect to the first matrix input
    :param X:
    :param W:
    :param sigma:
    :return:
    """

    assert X.shape[0] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # sum all the elements
    L = np.sum(S)

    # note: I'll refer to the derivatives by their quantities here,
    # unlike the math, where we referred to their function names

    # dLdS - just 1s
    dLdS = np.ones_like(S)

    # dSdN
    dSdN = deriv(sigma, N)

    # dLdN
    #dLdN = np.dot(dSdN, dLdS)
    dLdN = dLdS * dSdN

    # dNdX
    dNdX = np.transpose(W, (1, 0))

    # dLdX
    dLdX = np.dot(dSdN, dNdX)

    return dLdX