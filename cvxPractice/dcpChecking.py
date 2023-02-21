import cvxpy as cp
import numpy as np

x = cp.Variable()
y = cp.Variable()

# DCP Problems
prob1 = cp.Problem(cp.Minimize(cp.square(x - y)), [x + y >= 0])
prob2 = cp.Problem(cp.Maximize(cp.sqrt(x - y)), [2 * x - 3 == y, cp.square(x) <= 2])

print("prob1 is DCP: ", prob1.is_dcp())
print("prob2 is DCP: ", prob2.is_dcp())


# Non-DCP Problems

# A non-DCP objective
obj = cp.Maximize(cp.square(x))
prob3 = cp.Problem(obj)

print("prob3 is DCP:", prob3.is_dcp())
print("Maximize(square(x)) is DCP: ", obj.is_dcp())

# A non-DCP constraint
prob4 = cp.Problem(cp.Minimize(cp.square(x)), [cp.sqrt(x) <= 2])

print("prob4 is DCP:", prob4.is_dcp())
print("sqrt(x) <= 2 is DCP:", (cp.sqrt(x) <= 2).is_dcp())

# prob3.solve()