import numpy as np
from numba.experimental import jitclass
from numba import types

spec = [
    ("n_rbf", types.int32),
    ("n_inputs", types.int32),
    ("n_outputs", types.int32),
    ("theta", types.float64[:]),
]


# @jitclass(spec)
class RBF:
    # RBF calculates the values of the radial basis function that determine
    # the release

    # Attributes
    # -------------------
    # numberOfRBF       : int
    #                     number of radial basis functions, typically 2 more
    #                     than the number of inputs
    # numberofInputs    : int
    # numberOfOutputs   : int
    # center            : list
    #                     list of center values from optimization
    # radius            : list
    #                     list of radius values from optimization
    # weights           : list
    #                     list of weights from optimization
    # out               : list
    #                     list of the same size as the number of RBFs that
    #                     determines the control policy

    def __init__(self, n_rbf, n_inputs, n_outputs):
        self.n_rbf = n_rbf
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.theta = None
        self.weights = None
        self.center = None
        self.radius = None

    def set_parameters(self, theta):
        self.theta = theta
        theta = self.theta.copy()
        theta = theta.reshape((-1, 4))
        centerradius = theta[::2]
        self.weights = theta[1::2]
        self.center = centerradius[:, ::2]
        self.radius = centerradius[:, 1::2]

        ws = self.weights.sum(axis=0)
        for i in [np.where(ws == i)[0][0] for i in ws if i > 10**-6]:
            self.weights[:, i] = self.weights[:, i] / ws[i]

    def format_output(self, output):
        return (self.weights * (output.reshape(self.n_rbf, 1))).sum(axis=0)

    # def rbf_control_law(self, rbf_input):
    #     self.set_parameters()


# @jitclass(spec)
class GaussianRBF(RBF):
    def rbf_control_law(self, rbf_input):
        output = np.exp(
            -(np.sum((self.radius * (rbf_input - self.center)) ** 2, axis=1))
        )
        output = self.format_output(output)
        return output


# @jitclass(spec)
class MultiquadricRBF(RBF):
    def rbf_control_law(self, rbf_input):
        output = np.sqrt(
            1 + np.sum((self.radius * (rbf_input - self.center)) ** 2, axis=1)
        )
        output = self.format_output(output)
        return output


# @jitclass(spec)
class Multiquadric2RBF(RBF):
    def rbf_control_law(self, rbf_input):
        output = np.sqrt(
            np.sum((self.radius**2) + ((rbf_input - self.center) ** 2), axis=1)
        )
        output = self.format_output(output)
        return output


# @jitclass(spec)
class InvmultiquadricRBF(RBF):
    def rbf_control_law(self, rbf_input):
        output = 1 / np.sqrt(
            1 + np.sum((self.radius * (rbf_input - self.center)) ** 2, axis=1)
        )
        output = self.format_output(output)
        return output


# @jitclass(spec)
class InvquadraticRBF(RBF):
    def rbf_control_law(self, rbf_input):
        output = 1 / (
            1 + np.sum((self.radius * (rbf_input - self.center)) ** 2, axis=1)
        )
        output = self.format_output(output)
        return output


# @jitclass(spec)
class ExponentialRBF(RBF):
    def rbf_control_law(self, rbf_input):
        output = np.exp(-(np.sum((rbf_input - self.center) / self.radius, axis=1)))
        output = self.format_output(output)
        return output


# @jitclass(spec)
class SquaredexponentialRBF(RBF):
    def rbf_control_law(self, rbf_input):
        output = np.exp(
            -(np.sum((rbf_input - self.center) ** 2 / (self.radius**2), axis=1))
        )
        output = self.format_output(output)
        return output


# @jitclass(spec)
class Matern32RBF(RBF):
    def rbf_control_law(self, rbf_input):
        output = (
            1 + np.sqrt(3 * np.sum((rbf_input - self.center) / self.radius, axis=1))
        ) * (
            np.exp(
                -np.sqrt(3 * np.sum((rbf_input - self.center) / self.radius, axis=1))
            )
        )

        output = self.format_output(output)
        return output
