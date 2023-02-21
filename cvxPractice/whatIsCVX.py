import cvxpy as cp

x = cp.Variable()
y = cp.Variable()

constraints = [x + y == 1, x - y >= 1]

obj = cp.Minimize((x - y) ** 2)

prob = cp.Problem(obj, constraints)
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)

# Replace objective
prob2 = cp.Problem(cp.Maximize(x + y), prob.constraints)
print("optimal value", prob2.solve())

# Replace the constraint (x + y == 1)
constraints = [x + y <= 3] + prob2.constraints[1:]
prob3 = cp.Problem(prob2.objective, constraints)
print("optimal value", prob3.solve())

# An infeasible problem
prob_inf = cp.Problem(cp.Minimize(x), [x >= 1, x <= 0])
prob_inf.solve()
print("status:", prob_inf.status)
print("optimal value", prob_inf.value)

# A scalar
a = cp.Variable()

# A Vector
x = cp.Variable(5)

# A Matrix
A = cp.Variable((4,7))


