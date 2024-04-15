import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        # Khởi tạo kích thước của các layer và ma trận trọng số và bias
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Khởi tạo ma trận trọng số và bias cho các hidden layer và output layer
        self.weights = []
        self.biases = []
        
        # Khởi tạo trọng số và bias cho hidden layer đầu tiên
        self.weights.append(np.random.rand(input_size, hidden_sizes[0]))
        self.biases.append(np.zeros(hidden_sizes[0]))
        
        # Khởi tạo trọng số và bias cho các hidden layer còn lại
        for i in range(len(hidden_sizes)-1):
            self.weights.append(np.random.rand(hidden_sizes[i], hidden_sizes[i+1]))
            self.biases.append(np.zeros(hidden_sizes[i+1]))
        
        # Khởi tạo trọng số và bias cho output layer
        self.weights.append(np.random.rand(hidden_sizes[-1], output_size))
        self.biases.append(np.zeros(output_size))
        
    def sigmoid(self, x):
        # Hàm kích hoạt sigmoid
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inputs):
        # Tính toán output của các hidden layer và output layer
        for i in range(len(self.weights)):
            inputs = np.dot(inputs, self.weights[i]) + self.biases[i]
            inputs = self.sigmoid(inputs)
        return inputs

# Định nghĩa kích thước của các layer
input_size = 4
hidden_sizes = [3, 4, 5, 6]
output_size = 1

# Khởi tạo neural network với 3 hidden layer
nn = NeuralNetwork(input_size, hidden_sizes, output_size)

# Tạo một input sample
inputs = np.array([0.5, 0.3, 0.2, 0.4])

# Thực hiện forward propagation
outputs = nn.forward(inputs)

# In ra output của neural network
print("Output của neural network:", outputs)
