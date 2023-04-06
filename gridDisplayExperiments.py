import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(-30, 30)
ys = np.linspace(0, 80)

xi, yi = np.meshgrid(xs, ys)

zi = np.linspace(0, 1, len(xs) * len(ys)).reshape(len(xs), len(ys)).T

plt.pcolormesh(xi, yi, zi)
plt.show()