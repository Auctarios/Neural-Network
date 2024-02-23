from Layer import *
from Activation import Activation
import numpy as np
class Dense(Layer):
    def __init__(self, size: int = 1, activation = "relu"):
        if not isinstance(size, int):
            raise TypeError(f"size should be integer, instead given size type: {type(size)}")
        if not isinstance(activation, str):
            if not isinstance(activation, Activation):
                raise TypeError(f"activation should be either string or Activation, instead given activation type: {type(activation)}")
            self.activation = activation.act
            self.activation_prime = activation.act_prime
        else:
            self.activation = Activation(activation).act
            self.activation_prime = Activation(activation).act_prime


        self.act_str = activation
        self.size = size
        self.weights = None
        self.bias = None


    def initialize(self, input_size):
        self.weights = np.random.rand(input_size, self.size) - 0.5
        print(self.weights.shape)
        self.bias = np.random.rand(1, self.size) - 0.5


    def forward(self, input):
        lin_out = np.dot(input, self.weights) + self.bias
        output = self.activation(lin_out)
        return output


    def forward_propagation(self, input):
        if self.weights is None or self.bias is None:
            raise ValueError("Layer weights not initialized.")
        self.input = input
        

        self.linear_output = np.dot(self.input, self.weights) + self.bias

        self.output = self.activation(self.linear_output)

        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        a = self.activation_prime(self.linear_output)
        b = output_error

        # print("-------------------")
        # print("a type", type(a))
        # print("b type", type(b))
        # print("a shape", a.shape)
        # print("b shape", b.shape)
        
        if b.ndim == 1:
            a = a.reshape(-1)
        
        # print("AAa type", type(a))
        # print("AAb type", type(b))
        # print("AAa shape", a.shape)
        # print("AAb shape", b.shape)

        error = a * b

        # print("error type", type(error))
        # print("error shape", error.shape)

        error = np.array(error)

        if error.ndim == 1:
            error = error.reshape(-1, 1)

        # print("AAerror type", type(error))
        # print("AAerror shape", error.shape)

        # print("-------------------")
        
        # error = self.activation_prime(self.linear_output) * output_error
        input_error = np.dot(error, self.weights.T)
        weights_error = np.dot(self.input.T, error) / self.input.shape[0]
        bias_error = np.mean(error, axis=0, keepdims=True)
        # print(f"Weights {self.weights}")
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error
        

        




