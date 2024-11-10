import numpy as np
from numpy.typing import ArrayLike

def mi(pr: ArrayLike, pet: ArrayLike) -> ArrayLike:
    return np.sum(pr)/np.sum(pet)