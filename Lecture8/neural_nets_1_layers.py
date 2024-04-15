import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Khởi tạo kích thước của các layer và ma trận trọng số và bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Khởi tạo ma trận trọng số cho input layer và hidden layer, và bias cho hidden layer
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)
        
        # Khởi tạo ma trận trọng số cho hidden layer và output layer, và bias cho output layer
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
        
        return hidden_inputs, hidden_outputs, final_inputs, final_outputs
        # return final_outputs

# Định nghĩa kích thước của các layer
input_size = 4
hidden_size = 3
output_size = 1

# Khởi tạo neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Tạo một input sample
inputs = np.array([0.5, 0.3, 0.2, 0.4])

# Thực hiện forward propagation
hidden_inputs, hidden_outputs, final_inputs, final_outputs = nn.forward(inputs)
# final_outputs = nn.forward(inputs)

# In ra output của neural network
print("Output của neural network:", final_outputs)