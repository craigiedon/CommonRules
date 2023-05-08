import matplotlib.pyplot as plt
import numpy as np

w = 3.0
c = 0.0

ys = np.linspace(-8, 8, 200)

slope = 0.5

fs = 1.0 / (1 + np.exp(slope * ((ys - c) ** 2 - w ** 2)) - np.exp(slope * -(w**2)))

plt.plot(ys, fs)
plt.show()