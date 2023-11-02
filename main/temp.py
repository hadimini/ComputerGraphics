####################################################################
##### DRAFT
# vecs of lambdas are not correct, try to solve then manually
from sympy import symbols, Eq, solve
# we need vecs only for eigenval_max, so no need to calculate the other vector
x1, x2 = symbols('x1,x2')
eq1 = Eq((M.item(0)-eigenval_max) * x1 + M.item(1) * x2, 0)
eq2 = Eq(M.item(2) * x1 + (M.item(3) - eigenval_max) * x2, 0)
print('\n ------INIT\n', a, b)
print('\n break \n')
result = solve((eq1,eq2), (x1, x2))
print(result)
a, b = np.abs(a), np.abs(b)
print('\n ------BB\n', a, b)
# solve the equation just to get the sign
result_x1 = result.get(x1, None)
result_x2 = result.get(x2, None)

# if not(result_x1 and result_x2):
#     a = -a

# if result_x1 and (result_x1.subs({x2:1})) < 0:
#     a = -a

print('\n ------AA\n', a, b)
##### END DRAFT
#################################################################