import numpy as np

# Tạo một ma trận numpy
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print("Ma trận ban đầu:")
print(matrix)

# Shape của ma trận
print("Ma trận có shape:")
print(matrix.shape)

# 1. Tính tổng các phần tử của ma trận
print("\n1. Tổng các phần tử của ma trận:")
print(np.sum(matrix))

# 2. Tính tổng các phần tử theo từng cột
print("\n2. Tổng các phần tử theo từng cột:")
print(np.sum(matrix, axis=0))

# 3. Tính tổng các phần tử theo từng hàng
print("\n3. Tổng các phần tử theo từng hàng:")
print(np.sum(matrix, axis=1))

# 4. Tính tích các phần tử của ma trận
print("\n4. Tích các phần tử của ma trận:")
print(np.prod(matrix))

# 5. Tính giá trị trung bình của ma trận
print("\n5. Giá trị trung bình của ma trận:")
print(np.mean(matrix))

# 6. Tính trung vị của ma trận
print("\n6. Trung vị của ma trận:")
print(np.median(matrix))

# 7. Tìm phần tử lớn nhất của ma trận
print("\n7. Phần tử lớn nhất của ma trận:")
print(np.max(matrix))

# 8. Tìm phần tử nhỏ nhất của ma trận
print("\n8. Phần tử nhỏ nhất của ma trận:")
print(np.min(matrix))

# 9. Reshape ma trận thành một ma trận khác có kích thước khác
print("\n9. Reshape ma trận:")
reshaped_matrix = np.reshape(matrix, (2, 6))
print(reshaped_matrix)

# 10. Chuyển vị ma trận
print("\n10. Ma trận chuyển vị:")
transposed_matrix = np.transpose(matrix)
print(transposed_matrix)
