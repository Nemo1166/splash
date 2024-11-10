import math
import numpy as np
from numpy.typing import ArrayLike

def dsin(x: ArrayLike):
    return np.sin(np.deg2rad(x))

def dcos(x: ArrayLike):
    return np.cos(np.deg2rad(x))

def julian_day(y: int, m: int, d: int):
    if m <= 2:
        y -= 1
        m += 12
    
    a = math.floor(y / 100)
    b = 2 - a + math.floor(a / 4)

    jde = math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5
    return jde
