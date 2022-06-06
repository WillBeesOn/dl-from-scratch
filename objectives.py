from math import log
from utils import mv_multi, transpose

# Contains objective functions at the end of a deep learning system


# Common static class for computing objective functions and gradients
class Objective:
    @staticmethod
    # Calculates the average evaluation for a batch of observations
    def eval(y, yhat, eval_feat):
        return sum(transpose(Objective.batch(y, yhat, eval_feat))[0]) / len(yhat)

    # Calculates batch of gradients. Basically same as eval but without the averaging.
    @staticmethod
    def gradient(y, yhat, grad_feat):
        return Objective.batch(y, yhat, grad_feat)

    # Common function to batch calculate some function across all observations
    @ staticmethod
    def batch(y, yhat, calc):
        return [Objective.calc_for_observation(y_obs, yhat_obs, calc) for (y_obs, yhat_obs) in zip(y, yhat)]

    # Calculates some function across features of a single observation
    @staticmethod
    def calc_for_observation(y_obs, yhat_obs, calc):
        return [calc(y, yhat) for (y, yhat) in zip(y_obs, yhat_obs)]


class LeastSquares:
    # Square the difference of expected and output value
    # Does batch mean evaluation
    @staticmethod
    def eval(y, yhat):
        return Objective.eval(y, yhat, LeastSquares.eval_feat)

    # Evaluates a single feature
    @staticmethod
    def eval_feat(y, yhat):
        return (y - yhat) ** 2

    # Batch gradient
    @staticmethod
    def gradient(y, yhat):
        return Objective.gradient(y, yhat, LeastSquares.gradient_feat)

    # An alias for gradient to be consistent with activation layers when layers are structured in a list
    @staticmethod
    def backward(y, yhat):
        return LeastSquares.gradient(y, yhat)

    # Gradient for a single feature
    @staticmethod
    def gradient_feat(y, yhat):
        return -2 * (y - yhat)


class LogLoss:
    # Compute negative likelihood, also known as the log loss
    # Batch eval
    @staticmethod
    def eval(y, yhat):
        return Objective.eval(y, yhat, LogLoss.eval_feat)

    @staticmethod
    def eval_feat(y, yhat, e=10**-10):
        return -(y * log(yhat + e) + (1 - y) * log(1 - yhat + e))

    # Return derivative of eval
    # Batch gradient
    @staticmethod
    def gradient(y, yhat):
        return Objective.gradient(y, yhat, LogLoss.gradient_feat)

    # An alias for gradient to be consistent with activation layers when layers are structured in a list
    @staticmethod
    def backward(y, yhat):
        return LogLoss.gradient(y, yhat)

    @staticmethod
    def gradient_feat(y, yhat, e=10**-10):
        return -(y - yhat) / (yhat * (1 - yhat) + e)


class CrossEntropy:
    # Compute cross entropy across 2 congruent distributions/vectors
    # log yhat elements and transpose at the same time, then multiply y with transposed log yhat
    # Batch eval. Since cross entropy is calculated per observation instead of feature, just call calc_for_observation.
    @staticmethod
    def eval(y, yhat):
        result = Objective.calc_for_observation(y, yhat, CrossEntropy.eval_feat)
        return sum(result) / len(y)

    # Calc cross entropy. Transposing yhat so it can be multiplied with y is building into adding stability constant.
    @staticmethod
    def eval_feat(y, yhat, e=10**-10):
        return -mv_multi(y, [[log(elm + e)] for elm in yhat])[0]

    # Batch gradient.
    # Since cross entropy is calculated per observation instead of feature, just call calc_for_observation.
    @staticmethod
    def gradient(y, yhat):
        return Objective.calc_for_observation(y, yhat, CrossEntropy.gradient_feat)

    # An alias for gradient to be consistent with activation layers when layers are structured in a list
    @staticmethod
    def backward(y, yhat):
        return CrossEntropy.gradient(y, yhat)

    @staticmethod
    def gradient_feat(y, yhat, e=10**-10):
        return [- pair[0] / pair[1] for pair in zip(y, [elm + e for elm in yhat])]
