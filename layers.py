from abc import ABC, abstractmethod
from math import exp, sqrt
from random import uniform
from utils import transpose, v_operation, m_operation, sm_operation, m_multi, mv_multi, sm_multi

# Contains layers in a deep learning system


# An abstract class representing a layer in a deep learning system
class Layer(ABC):
    def __init__(self):
        self.prev_input = []
        self.prev_output = []

    # Method used for forward propagation
    @abstractmethod
    def forward(self, data):
        pass

    # Method for getting this layer's gradient
    @abstractmethod
    def gradient(self):
        pass


# Input layer of deep learning system, responsible for accepting initial data
class InputLayer(Layer):
    def __init__(self, data):
        super().__init__()

        # Calculate the mean of each feature across the data set
        transposed = transpose(data)  # Transpose so each row contains all data of a single feature
        count = len(transposed[0])  # Count of data points
        self.meanX = [sum(feat) / count for feat in transposed]

        # Calculate standard deviation of each feature across the data set
        self.stdX = [sqrt(sum([(n - self.meanX[i]) ** 2 for n in feat]) / count) for i, feat in enumerate(transposed)]
        self.stdX = [1 if x == 0 else x for x in self.stdX]  # If a standard deviation is 0, set it to 1 instead

    # Calculate zscore for data and return it.
    def forward(self, data):
        if not isinstance(data, list):
            return

        self.prev_input = data

        # Calculate zscore for data. # of standard deviations away data is from the mean.
        self.prev_output = [[(feat - self.meanX[i]) / self.stdX[i] for i, feat in enumerate(row)] for row in data]
        return self.prev_output

    # Required to have here
    def gradient(self):
        pass


# Linear layer.
class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    # Input = output. Do nothing with input and simply return it.
    def forward(self, data):
        if not isinstance(data, list):
            return

        self.prev_input = data
        self.prev_output = data
        return self.prev_output

    # Derive prev_output wrt prev_input. Derivative of linear is 1, and only when output and input indices match.
    # Number of inputs = number of outputs so for each row, return vector of 1s of length of input/outputs
    def gradient(self):
        if len(self.prev_output) == 0:
            return
        row_len = len(self.prev_output[0])
        a = [[1 for i in range(0, row_len)] for row in self.prev_input]
        return a

    # Remember that the linear gradient is computed as vectors of the diagonals of the actual gradient.
    # Be sure to do the Hadamard product when computing this (not that computing this really matters for linear anyway).
    def backward(self, grad_in):
        return m_operation('*', grad_in, self.gradient())


# Rectified Linear Unit layer.
class ReLuLayer(Layer):
    def __init__(self):
        super().__init__()

    # Allows positive values, zeroes out negative values.
    def forward(self, data):
        if not isinstance(data, list):
            return

        self.prev_input = data

        # Set values to 0 if it's below 0. Otherwise, leave data alone. Record output.
        self.prev_output = [[max(0, feat) for i, feat in enumerate(row)] for row in data]
        return self.prev_output

    # Derive prev_output wrt prev_input. Derivative of ReLu is 1 when z > 0, 0 when z <= 0. Only across diagonal.
    def gradient(self):
        if len(self.prev_output) == 0:
            return
        return [[1 if z > 0 else 0 for z in row] for row in self.prev_input]

    def backward(self, grad_in):
        return m_operation('*', grad_in, self.gradient())


# Logistic Sigmoid layer
class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    # Keep values in range of (0, 1)
    def forward(self, data):
        if not isinstance(data, list):
            return

        self.prev_input = data

        # Calculate and record output.
        self.prev_output = [[self.__calc(feat) for feat in row] for row in data]
        return self.prev_output

    # Derive prev_output wrt prev_input. Derivative of logistic sigmoid g(z) * (1 - g(z)) + e along diagonal.
    # e (epsilon) is a numeric stability constant to prevent issues with logarithms when output is 0.
    def gradient(self, e=10 ** -10):
        if len(self.prev_output) == 0:
            return
        return [[self.__calc(z) * (1 - self.__calc(z)) + e for z in row] for row in self.prev_input]

    def backward(self, grad_in):
        return m_operation('*', grad_in, self.gradient())

    # Calculate the logistic sigmoid given a number.
    @staticmethod
    def __calc(num):
        return 1 / (1 + exp(-num))


# Softmax layer
class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    # Calculates probability distribution for each piece of data.
    # Each feature is in the range [0, 1] and all features in each observation sum to 1
    def forward(self, data):
        if not isinstance(data, list):
            return

        self.prev_input = data

        # Calculate and record data.
        new_output = [self.__calc(row) for row in data]
        self.prev_output = new_output
        return self.prev_output

    # Since each output relies on every input, the gradient matrix for each observation is not filled in only along
    # the diagonal. We need to create the full matrix and not just a vector representing the diagonal.
    # Diagonal = gj(z) * (1 - gj(z))   Other = -gj(z) * gi(z)
    def gradient(self):
        if len(self.prev_output) == 0:
            return
        feats = len(self.prev_input[0])
        all_gradients = []  # Gradients for each observation

        # For each observation, create a gradient matrix
        for row in self.prev_input:
            exps = self.__calc_feat_exps(row)  # Calc exponents for all features in row
            exp_sum = sum(exps)  # Sum them all for denominator
            gradient_for_obsv = []  # Gradient matrix for a single observation

            # For each feature...
            for j in range(0, feats):
                gj = (exps[j] / exp_sum)  # Calculate softmax for a single feature
                gradient_row = []

                # ...change the feature by which to derive.
                for i in range(0, feats):
                    if i == j:
                        gradient_row.append(gj * (1 - gj))  # Derive feature wrt itself
                    else:
                        gi = exps[i] / exp_sum
                        gradient_row.append(-gj * gi)

                gradient_for_obsv.append(gradient_row)
            all_gradients.append(gradient_for_obsv)
        return all_gradients

    # Matrix multiplication gradient calculation.
    def backward(self, grad_in):
        return [mv_multi(g_in, layer_g) for (g_in, layer_g) in zip(grad_in, self.gradient())]

    # Calculate the softmax of a single observation.
    @classmethod
    def __calc(cls, row):
        exps = cls.__calc_feat_exps(row)
        exp_sum = sum(exps)
        return [feat_exp / exp_sum for feat_exp in exps]

    # Calculate the parameters used in calculating softmax for a single observation.
    @staticmethod
    def __calc_feat_exps(row):
        feat_max = max(row)  # Subtract max feature value to avoid under/overflow
        feat_exps = [exp(feat - feat_max) for feat in row]  # Compute exponents for each feature
        return feat_exps


# Hyperbolic Tangent layer
class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    # Similar to sigmoid, but clamps value range to (-1, 1)
    def forward(self, data):
        if not isinstance(data, list):
            return

        self.prev_input = data

        # Calculate and record data.
        self.prev_output = [[self.__calc(feat) for feat in row] for row in data]
        return self.prev_output

    # Derive prev_output wrt prev_input. Derivative of logistic sigmoid (1 - g(z)^2) + e along diagonal.
    # e (epsilon) is a numeric stability constant to prevent issues with logarithms when output is 0.
    def gradient(self, e=10 ** -10):
        if len(self.prev_output) == 0:
            return
        return [[1 - self.__calc(z) ** 2 + e for z in row] for row in self.prev_input]

    def backward(self, grad_in):
        return m_operation('*', grad_in, self.gradient())

    # Calculate hyperbolic tangent of a number.
    @staticmethod
    def __calc(num):
        exp_num = exp(num)
        neg_exp_num = exp(-num)
        return (exp_num - neg_exp_num) / (exp_num + neg_exp_num)


class FullyConnected(Layer):
    def __init__(self, input_feats, output_feats, bias=[], w_range=0.00001):
        super().__init__()

        # If bias is not specified, then initialize a zero vector to match the number of output features.
        if len(bias) == 0:
            bias = [0 for i in range(0, output_feats)]

        # Initialize random weights in range [-10^-4, 10^-4]
        # Height = # of input feats. Width = # of output feats. Input rows are being multiplied by the weight matrix.
        self.weights = [[uniform(-w_range, w_range) for j in range(0, output_feats)] for i in range(0, input_feats)]

        # Set bias from parameter.
        self.bias = bias

        # Accumulators for ADAM adaptive learning. Initialize as 0 for each weight.
        self.adam_momentum_acc = [[0 for j in range(0, output_feats)] for i in range(0, input_feats)]
        self.adam_scale_acc = [[0 for j in range(0, output_feats)] for i in range(0, input_feats)]

    # Compute new values using weights and bias
    def forward(self, data):
        if not isinstance(data, list):
            return

        self.prev_input = data

        # Multiply data row by weight matrix, then add bias.
        self.prev_output = [v_operation('+', mv_multi(row, self.weights), self.bias) for row in data]

        return self.prev_output

    # Deriving FCL just returns weights transposed
    def gradient(self):
        if len(self.prev_output) == 0:
            return
        return transpose(self.weights)

    def backward(self, grad_in):
        self.updateWeights(grad_in)
        return m_multi(grad_in, self.gradient())

    # Updates the fully connected layer's weights and biases
    # w_dir and b_dir used to determine if we are adding or subtracting the weight and bias increments
    def updateWeights(self, grad_in, l_rate=0.0001, adam=False, e=10 ** -10,
                      p1=0.9, p2=0.999, epoch=1, w_dir=-1, b_dir=-1):
        djdw = m_multi(transpose(self.prev_input), grad_in)  # dj/dw = observation^T * dj/dh

        if adam:
            self.adam_momentum_acc = m_operation(
                '+',
                sm_operation('*', p1, self.adam_momentum_acc),
                sm_operation('*', 1 - p1, djdw)
            )
            self.adam_scale_acc = m_operation(
                '+',
                sm_operation('*', p2, self.adam_scale_acc),
                sm_operation('*', 1 - p2, m_operation('*', djdw, djdw))
            )

            momentum = sm_operation('*', 1 / (1 - p1 ** epoch), self.adam_momentum_acc)
            scale = [[sqrt(val / (1 - p2 ** epoch)) + e for val in row] for row in self.adam_scale_acc]
            weight_increments = sm_operation('*', w_dir * l_rate / len(grad_in), m_operation('/', momentum, scale))
            self.weights = m_operation('+', self.weights, weight_increments)
        else:
            # scalar = dir * l_rate / data length
            increment_scalar = (w_dir * l_rate) / len(grad_in)
            weight_increment = sm_multi(increment_scalar, djdw)  # increment = scalar * dj/dw
            self.weights = m_operation('+', self.weights, weight_increment)  # new = old + increment

        # Due to rounding errors during backprop, dj/db may not be entirely accurate.
        # Find the average of each feature to get a more exact bias increment.
        transposed_bias_increment = transpose(sm_multi(b_dir * l_rate, grad_in))
        new_bias_increment = [sum(vec) / len(vec) for vec in transposed_bias_increment]
        self.bias = v_operation('+', self.bias, new_bias_increment)
