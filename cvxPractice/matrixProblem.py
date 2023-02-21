import cvxpy as cp
import numpy

m = 10
n = 5
numpy.random.seed(1)

A = numpy.random.randn(m,n)
b = numpy.random.randn(m)

x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)