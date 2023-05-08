import time

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

T = 60  # Timesteps
dt = 0.1

# Z-State
x = cp.Variable(T)
y = cp.Variable(T)
vx = cp.Variable(T)
vy = cp.Variable(T)

# U-Actions
ax = cp.Variable(T)
ay = cp.Variable(T)

# v = cp.Variable(T)
# h = cp.Variable(T)

# Initial State
x_0 = 0.0
y_0 = 0.0
vx_0 = 0.0
vy_0 = 0.0

# Goal Position
x_g = 200
y_g = 0

# Max velocity
# v_max = 10


w_gx = 1.0
w_gy = 1.0
w_ga = 1.0

# prog_x = cp.sum((x - x_g) ** 2)
prog_x = cp.sum_squares(x - x_g)
deviation_y = cp.sum_squares(y - y_g)
min_lat_accs = cp.sum_squares(ay)

obj = cp.Minimize(w_gx * prog_x + w_gy * deviation_y + w_ga * min_lat_accs) #  + w_ga * min_lat_accs)

# obj = cp.Minimize(cp.sum(x))


start_constraint = [
    x[0] == x_0,
    y[0] == y_0,
    vx[0] == vx_0,
    vy[0] == vy_0
]

f_dyns = [
    x[1:] == x[:-1] + dt * vx[:-1],
    y[1:] == y[:-1] + dt * vy[:-1],
    vx[1:] == vx[:-1] + dt * ax[:-1],
    vy[1:] == vy[:-1] + dt * ay[:-1]
]

prop_vels = (vx >= 1.5 * cp.abs(vy))

ax_min, ax_max = -11.5, 11.5
ay_min, ay_max = -2.0, 2.0

acc_lims = [
    ax_min <= ax,
    ax <= ax_max,
    ay_min <= ay,
    ay <= ay_max
]

vx_min, vx_max = 0, 45
vy_min, vy_max = -10, 10

vel_lims = [
    vx_min <= vx,
    vx <= vx_max,
    vy_min <= vy,
    vy <= vy_max
]

# jerk_lims = [
#     axj_min <= 0 <= axj_max
#     ayj_min <= 0 <= ayj_max
# ]

obs_xs = np.ones(T) * 10.0
obs_ys = np.ones(T) * 0.0
obs_l = 4.0
obs_w = 1.5

ego_l = 4.0
ego_w = 1.5

obs_xmin = obs_xs - (obs_l / 2.0) - ego_l / 2.0
obs_xmax = obs_xs + (obs_l / 2.0) + ego_l / 2.0
obs_ymin = obs_ys - (obs_w / 2.0) - ego_w / 2.0
obs_ymax = obs_ys + (obs_w / 2.0) + ego_w / 2.0

M = 1000

# mu_k = (cvxpy.maximum(0.0, obs_xmin - x) +
#         cvxpy.maximum(x - obs_xmax, 0.0) +
#         cvxpy.maximum(obs_ymin - y, 0.0))
# mu_k = cvxpy.maximum(obs_xmin[0] - x[0], 0.0) # cvxpy.maximum(obs_xmin - x, 0.0)

# b = cp.Variable(T, boolean=True)

b = cp.Variable((T, 4), boolean=True)
# b1 = cp.Variable(T, boolean=True)
# b2 = cp.Variable(T, boolean=True)
# b3 = cp.Variable(T, boolean=True)
# b4 = cp.Variable(T, boolean=True)



# b1_c = obs_xmin - x <= M * b
# b2_c = (x - obs_xmax) <= M * b
# b3_c = (obs_ymin - y) <= M * b

# If x is between obs xmin and obs xmax and y is greater than obs_ymin
# So...either:

# x < oxmin, x > oxmar, or y < ymin OR
# you have to set y > ymax

# box_bounds = cp.vstack([x - obs_xmin, obs_xmax - x, y - obs_ymin, obs_ymax - y]).T
# disj_cs = [
#     box_bounds <= M * b,
#     cp.sum(b, axis=1) <= 3
# ]

disj_cs = [
    x <= obs_xmin + M * b[:, 0],
    obs_xmax <= x + M * b[:, 1],
    y <= obs_ymin + M * b[:, 2],
    obs_ymax <= y + M * b[:, 3],
    cp.sum(b, axis=1) <= 3
]
# obs_avoid_const = obs_ymax - M * (1 - b) <= y


constraints = [
    *start_constraint,
    prop_vels,
    *acc_lims,
    *vel_lims,
    *f_dyns,
    # obs_avoid_const,
    # b1_c,
    # b3_c
    # b1_c, b2_c, b3_c,
    # b1_c
    *disj_cs
]

prob = cp.Problem(obj, constraints)

print(cp.installed_solvers())

start_time = time.time()
result = prob.solve(solver="SCIP")
print("Solve Time: ", time.time() - start_time)

print("Obj func: ", result)
# print("X Vals:", x.value)
# print("Y Vals:", y.value)

plt.subplot(211)
plt.plot(range(T), x.value)
plt.xlabel("T")
plt.ylabel("X")

plt.subplot(212)
plt.plot(range(T), y.value)
plt.xlabel("T")
plt.ylabel("Y")

# TODO: Plot the actual car!
# TODO: Actually just draw and animate at this point?
# TODO: Make sure to not do something silly like restrict things that are impossible on turn 0?


print("B: ", b.value)
# print("B1: ", b1.value)
# print("B2: ", b2.value)
# print("B3: ", b3.value)
# print("B4: ", b4.value)

plt.show()
