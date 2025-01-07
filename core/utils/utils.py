import numpy as np
from numba import njit

LIST_SPECIFIC_CHARACTERS = "[],"


def generate_random_actions(number_of_actions=4, seed=42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.rand(4) * [10000, 10000, 10000, 4000]


def convert_str_to_float_list(string_list: str) -> list:
    return list(map(float, string_list.translate(str.maketrans("", "", LIST_SPECIFIC_CHARACTERS)).split()))

@njit
def gallonToCubicFeet(x):
    conv = 0.13368  # 1 gallon = 0.13368 cf
    return x * conv


@njit
def inchesToFeet(x):
    conv = 0.08333  # 1 inch = 0.08333 ft
    return x * conv


@njit
def cubicFeetToCubicMeters(x):
    conv = 0.0283  # 1 cf = 0.0283 m3
    return x * conv


@njit
def feetToMeters(x):
    conv = 0.3048  # 1 ft = 0.3048 m
    return x * conv


@njit
def acreToSquaredFeet(x):
    conv = 43560  # 1 acre = 43560 feet2
    return x * conv


@njit
def acreFeetToCubicFeet(x):
    conv = 43560  # 1 acre-feet = 43560 feet3
    return x * conv


@njit
def cubicFeetToAcreFeet(x):
    conv = 43560  # 1 acre = 43560 feet2
    return x / conv


@njit
def interpolate_tailwater_level(X, Y, x):
    dim = len(X) - 1
    if x <= X[0]:
        y = (x - X[0]) * (Y[1] - Y[0]) / (X[1] - X[0]) + Y[0]
        return y
    elif x >= X[dim]:
        y = Y[dim] + (Y[dim] - Y[dim - 1]) / (X[dim] - X[dim - 1]) * (
            x - X[dim]
        )
        return y
    else:
        y = np.interp(x, X, Y)
    return y

