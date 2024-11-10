import numpy as np
from numpy.typing import ArrayLike

# Reference: https://doi.org/10.1098/rspa.2020.0346
def Alpha(mi: ArrayLike, omega: float=3):
    # if mi == 0:
    #     return np.nan
    return 1.26*mi*(1 + 1/mi - (1 + (1/mi)**omega)**(1/omega))
