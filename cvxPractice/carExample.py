import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

T = 20 # Timesteps

x = cp.Variable(T)
y = cp.Variable(T)
v = cp.Variable(T)
h = cp.Variable(T)

# Initial State
x_0 = 0.0
y_0 = 0.0
v_0 = 0.0

# Goal Position
x_g = 200
y_g = 5

# Max velocity
v_max = 10


w_gx = 1.0
w_gy = 1.0

obj = cp.Minimize(w_gx * cp.sum_squares(x - x_g) + w_gy * cp.sum_squares(y - y_g))


constraints = [
    x[0] == x_0,
    y[0] == 0,
    v <= v_max,
    h >= 0.0,
    h <= 2 * np.pi,
    x[1:] == x[:-1] + v[:-1] * np.cos(h),
    y[1:] == y[:-1] + v[:-1] * np.sin(h)]

prob = cp.Problem(obj, constraints)

result = prob.solve()

print("Obj func: ", result)
print("X Vals:", x.value)
print("Y Vals:", y.value)

plt.plot(x.value, y.value)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()