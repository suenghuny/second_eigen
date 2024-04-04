# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.ticker as ticker
#
# x_list = list()
# x_prime_list = list()
# y_list = list()
# z_list = list()
#
# for t in range(300000):
#     n = 25
#     w = np.random.uniform(0, 1)
#     x = w*np.ones(n).reshape([-1, 1])
#     A = x @ x.T
#     A = A - np.diag(np.diag(A))
#     D = np.diag(np.sum(A, axis=1))
#     L = D-A
#     F = np.sum(A*A)
#     Var = np.mean(np.var(A, axis = 1))
#     second_eig = np.linalg.eigh(L)[0][1]
#     print(second_eig, n/(n-1)*(F- n**2 *Var)**0.5)
#
#
# x_list_high = list()
# y_list_high = list()
# z_list_high = list()
# for i in range(0, 1000):
#     A = np.ones((n,n))*i/1000
#     A = A - np.diag(np.diag(A))
#     if np.random.uniform(0,1) <=0.5:
#         A[-1, :] = 0
#         A[:, -1] = 0
#         D = np.diag(np.sum(A, axis=1))
#         L = D - A
#         F = np.sum(A * A)  # frobenius nor
#         F_prime = np.sum(A)  # frobenius nor
#         Var = np.mean(np.var(A, axis=1))
#         second_eig = np.linalg.eigh(L)[0][1]
#         x_list_high.append(F)
#         y_list_high.append(Var)
#         z_list_high.append(second_eig)
#
#     D = np.diag(np.sum(A, axis=1))
#     L = D-A
#     F = np.sum(A*A) # frobenius nor
#     F_prime = np.sum(A) # frobenius nor
#     Var = np.mean(np.var(A, axis = 1))
#     second_eig = np.linalg.eigh(L)[0][1]
#     x_list.append(F)
#     x_prime_list.append(F_prime)
#     y_list.append(Var)
#     z_list.append(second_eig)
#

import numpy as np
x_list_high = list()
y_list_high = list()
for t in range(300000):
    n = 25
    x = np.random.uniform(0, 1, n).reshape([-1, 1])
    A = x @ x.T
    y = np.random.uniform(0, 1, n).reshape([-1, 1])
    B = y @ y.T
    deg_A = np.su
