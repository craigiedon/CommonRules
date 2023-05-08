import cvxpy as cp
import numpy as np

m, n = 2, 3

A = np.random.rand(m, n)
b = np.random.randn(m)

x = cp.Variable(n, integer=True)

objective = cp.Minimize(cp.sum_squares(A @ x - b))
prob = cp.Problem(objective)
prob.solve()

print("Status: ", prob.status)
print("The optimal value is: ", prob.value)
print("A solution is: ", x.value)
