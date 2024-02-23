import numpy as np
class Activation():

    def __init__(self, type: str):
        if not isinstance(type, str):
            raise TypeError("Given type is not string.")
        
        type = type.lower()

        avail_types = ["relu", "sigmoid", "tanh", "none", "lrelu"]

        if type not in avail_types:
            raise ValueError(f"Given activation function is not available, available activations are: {avail_types}")

        self.type = type

        self.act = getattr(self, type)
        type_prime = type + "_prime"
        self.act_prime = getattr(self, type_prime)


    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_prime(self, x):
        return 1 - (self.tanh(x)) ** 2
    
    def none(self, x):
        return x
    
    def none_prime(self, x):
        return 1
    
    def lrelu(self, x):
        x[x < 0] = x[x < 0]*0.01
        return x
    
    def lrelu_prime(self, x):
        x[x < 0] = 0.01
        x[x > 0] = 1
        return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x):
        # return [[i if i > 0 else 0 for i in x[j]] for j in x]
        return np.maximum(0,x)

    def relu_prime(self, x):
        return (x>0).astype(x.dtype)



