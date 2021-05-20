# Adds link to the scripts folder
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def sum(a: np.array, b: float) -> float :
    return np.sum(a) + b

if __name__ == "__main__":
    a = np.array([1,2,3])
    b = np.array([1])
    print(sum(a, b))
