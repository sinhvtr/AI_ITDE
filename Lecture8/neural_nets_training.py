import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Khởi tạo kích thước của các layer và ma trận trọng số và bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Khởi tạo ma trận trọng số và bias cho hidden layer và output layer
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)
        
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros(output_size)
        
    def sigmoid(self, x):
        # Hàm kích hoạt sigmoid
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inputs):
        # Tính toán output của hidden layer
        hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)
        
        # Tính toán output của output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_output
        final_outputs = self.sigmoid(final_inputs)
        
        return final_outputs
    
    def train(self, inputs, targets, learning_rate):
        # Forward propagation
        hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_output
        final_outputs = self.sigmoid(final_inputs)
        
        # Backpropagation
        output_errors = targets - final_outputs
        output_delta = output_errors * (final_outputs * (1 - final_outputs))
        
        hidden_errors = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_errors * (hidden_outputs * (1 - hidden_outputs))
        
        # Update weights and biases
        self.weights_hidden_output += np.dot(hidden_outputs.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate
        
# Dữ liệu huấn luyện
inputs = np.random.rand(100, 5)
targets = np.random.randint(0, 2, (100, 1))

# Thiết lập kích thước của các layer
input_size = 5
hidden_size = 3
output_size = 1

# Khởi tạo neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Huấn luyện neural network
epochs = 100
learning_rate = 0.001

for epoch in range(epochs):
    nn.train(inputs, targets, learning_rate)

# Kiểm tra kết quả
predictions = nn.forward(inputs[0])
print("Predictions after training:")
print(predictions)
