import numpy as np

# Thực hiện phép cộng hai ma trận
print("\nPhép cộng hai ma trận:")
matrix_a = np.random.rand(4, 3)
print('Matrix A: ', matrix_a)
matrix_b = np.random.rand(4, 3)
print('Matrix B: ', matrix_b)
result_sum = matrix_a + matrix_b
print('Tổng 2 ma trận: ')
print(result_sum)

# Thực hiện phép nhân ma trận
print("\nPhép nhân hai ma trận:")
matrix_c = np.random.rand(4, 3)
matrix_d = np.random.rand(3, 2)
result_product = np.dot(matrix_c, matrix_d)
print(result_product)

# Thực hiện phép nhân ma trận với một scalar
print("\nPhép nhân ma trận với scalar:")
scalar = 2
result_scalar_product = scalar * matrix_a
print(result_scalar_product)
