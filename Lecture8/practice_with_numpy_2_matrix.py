import numpy as np

# Khởi tạo một ma trận ngẫu nhiên
print("Khởi tạo ma trận ngẫu nhiên:")
random_matrix = np.random.rand(3, 3)
print(random_matrix)

# Khởi tạo một ma trận có kích thước 3x3 toàn giá trị 0
print("\nKhởi tạo ma trận 0:")
zeros_matrix = np.zeros((3, 3))
print(zeros_matrix)

# Khởi tạo một ma trận có kích thước 3x3 toàn giá trị 1
print("\nKhởi tạo ma trận 1:")
ones_matrix = np.ones((3, 3))
print(ones_matrix)

# Thực hiện phép cộng hai ma trận
print("\nPhép cộng hai ma trận:")
matrix_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix_b = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
result_sum = matrix_a + matrix_b
print(result_sum)

# Thực hiện phép trừ hai ma trận
print("\nPhép trừ hai ma trận:")
result_diff = matrix_a - matrix_b
print(result_diff)

# Thực hiện phép nhân ma trận
print("\nPhép nhân hai ma trận:")
result_product = np.dot(matrix_a, matrix_b)
print(result_product)

# Thực hiện phép nhân ma trận với một scalar
print("\nPhép nhân ma trận với scalar:")
scalar = 2
result_scalar_product = scalar * matrix_a
print(result_scalar_product)
