import numpy as np
class Activation():
    """
    A class to represent different activation functions for use in neural networks.

    Attributes
    ----------
    type : str
        The type of activation function to use. Supported activations are 'relu', 'sigmoid', 'tanh', 'none', 'lrelu', and 'softmax'.
    """
    def __init__(self, type: str):
        """
        Initializes the Activation object with the specified activation function type.

        Parameters
        ----------
        type : str
            The type of activation function to use.

        Raises
        ------
        TypeError
            If the type is not a string.
        ValueError
            If the specified activation function type is not supported.
        """
        if not isinstance(type, str):
            raise TypeError("Given type is not string.")
        
        type = type.lower()

        avail_types = ["relu", "sigmoid", "tanh", "none", "lrelu", "softmax"]

        if type not in avail_types:
            raise ValueError(f"Given activation function is not available, available activations are: {avail_types}")

        self.type = type

        self.act = getattr(self, type)
        type_prime = type + "_prime"
        self.act_prime = getattr(self, type_prime)


    def tanh(self, x):
        """
        Computes the hyperbolic tangent of 'x'.
        """
        return np.tanh(x)
    
    def tanh_prime(self, x):
        """
        Computes the derivative of the hyperbolic tangent function.
        """
        return 1 - (self.tanh(x)) ** 2
    
    def none(self, x):
        """
        Returns the input without any modifications.
        """
        return x
    
    def none_prime(self, x):
        """
        Returns the derivative of the 'none' activation function, which is simply ones.
        """
        return np.ones_like(x)
    
    def lrelu(self, x):
        """
        Computes the leaky ReLU activation function.
        """
        x[x < 0] = x[x < 0]*0.01
        return x
    
    def lrelu_prime(self, x):
        """
        Computes the derivative of the leaky ReLU activation function.
        """
        x[x < 0] = 0.01
        x[x > 0] = 1
        return x

    def sigmoid(self, x):
        """
        Computes the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        """
        Computes the derivative of the sigmoid activation function.
        """
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x):
        """
        Computes the Rectified Linear Unit (ReLU) activation function.
        """
        return np.maximum(0,x)

    def relu_prime(self, x):
        """
        Computes the derivative of the ReLU activation function.
        """
        return (x>0).astype(x.dtype)
    
    def softmax(self, x):
        """
        Computes the softmax activation function.
        """
        e_x = np.exp(x - np.max(x, axis = 1, keepdims=True))
        return e_x / np.sum(e_x, axis = 1, keepdims=True)
    
    def softmax_prime(self, x):
        """No need."""
        pass



