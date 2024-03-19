from Layer import *
from Activation import Activation
from Optimizers import *
import numpy as np
class Dense(Layer):
    """
    Represents a dense (fully connected) layer within a neural network.

    Attributes
    ----------
    size : int
        The number of neurons in the layer.
    activation : str or Activation
        The activation function to be used in the layer. Can be specified by name as a string or by passing an Activation object.
    dropout_rate : float
        The rate at which neurons are randomly ignored during training to prevent overfitting. Must be between 0 and 1.
    optimizer : Optimizer
        The optimization algorithm used to update the layer's parameters during training.
    weights : np.ndarray
        The weights matrix of the layer. Each row represents the weights connecting a neuron in the previous layer to this layer.
    bias : np.ndarray
        The bias vector of the layer. Each element is the bias for a neuron in the layer.
    """
    def __init__(self, size: int = 1, activation = "relu", dropout_rate = 0.0):
        """
        Initializes the Dense layer with the specified size, activation function, and dropout rate.

        Parameters
        ----------
        size : int, optional
            The number of neurons in the layer (default is 1).
        activation : str or Activation, optional
            The activation function to be used in the layer. Can be a string name of the function or an Activation object (default is "relu").
        dropout_rate : float, optional
            The dropout rate, representing the proportion of neurons to drop during training to prevent overfitting (default is 0.0).
        
        Raises
        ------
        ValueError
            If the dropout_rate is not within the range [0.0, 1.0).
        TypeError
            If the provided size is not an integer, or if the activation is neither a string nor an Activation object.
        """
        if dropout_rate < 0.0 or dropout_rate >= 1.0:
            raise ValueError(f"dropout_rate should be in range [0.0, 1) instead given dropout_rate is {dropout_rate}")
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

        self.dropout_rate = dropout_rate
        self.optimizer = None
        self.act_str = activation
        self.size = size
        self.weights = None
        self.bias = None


    def initialize(self, input_size, optimizer):
        """
        Initializes the layer's weights and biases based on the size of the input and the specified optimizer.
        """
        if self.weights is None:
            self.weights = np.random.randn(input_size, self.size) * np.sqrt(2/self.size)
            self.bias = np.zeros((1, self.size)) 
        if self.optimizer is None:
            self.optimizer = optimizer
        

    def forward(self, input):
        """
        Computes the forward pass of the layer using the input data.
        """
        lin_out = np.dot(input, self.weights) + self.bias
        output = self.activation(lin_out)
        return output
    



    def forward_propagation(self, input):
        """
        Performs forward propagation through this layer, applying the activation function and dropout if specified.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Layer weights not initialized.")
        self.input = input
        
        #normal output
        self.linear_output = np.dot(self.input, self.weights) + self.bias
        #after activation
        self.output = self.activation(self.linear_output)

        if self.dropout_rate > 0.0:
            self.output *= np.random.binomial(1, 1-self.dropout_rate, size=self.output.shape)

        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        """
        Performs backward propagation through tihs layer, computing the gradients and updating the weights and biases.
        """
        if self.act_str == "softmax":
            error = output_error
        else:
            a = self.activation_prime(self.linear_output)
            b = output_error
            if b.ndim == 1:
                a = a.reshape(-1)

            error = a * b      

        error = np.array(error)

        if error.ndim == 1:
            error = error.reshape(-1, 1)

        input_error = np.dot(error, self.weights.T)

        weights_error = np.dot(self.input.T, error) / self.input.shape[0]
        bias_error = np.mean(error, axis=0, keepdims=True)    

        self.weights, self.bias = self.optimizer.update(self.weights, weights_error,
                                                        self.bias, bias_error, learning_rate)

        return input_error
        
