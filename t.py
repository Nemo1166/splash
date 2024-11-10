import numpy as np

import matplotlib.pyplot as plt

lat = np.linspace(-90, 90, 100)
lon = np.linspace(-180, 180, 200)

data = np.random.rand(100, 200)

plt.contour(lon, lat, data, bins=20)
plt.show()