import sys
from Layer import *
import warnings
import numpy as np
from Losses import Losses
class Sequential:
    def __init__(self):
        self.layers = []
        self.errors = []
        self.val_errors = None

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError(f"Given layer is {type(layer)}, instead it should be Layer")
        
        self.layers.append(layer)

    def build(self, input_shape):
        prev_output_size = input_shape
        for layer in self.layers:
            if layer.weights is None:
                layer.initialize(prev_output_size)
            prev_output_size = layer.size

    def forward_propagation(self, input_data):
        for layer in self.layers:
            input_data = layer.forward_propagation(input_data)
        return input_data
    
    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data
    
    def backward_propagation(self, output_error, learning_rate):
        for layer in reversed(self.layers):
            output_error = layer.backward_propagation(output_error, learning_rate)

    def train(self, x_train, y_train, epochs, learning_rate, batch_size = 32, val_set: tuple = None, loss: str = "MSE", verbose: int = 5):
        self.val_errors = [] if val_set is not None else None
        looss = Losses(loss)
        self.loss = looss.loss
        self.loss_derivative = looss.loss_derivative
        print(x_train.shape)
        self.build(x_train.shape[1])
        if val_set is not None:
            val_err = -1
        group = len(x_train)//batch_size

        for i in range(epochs):
            # if i > 5:
            #     learning_rate -= i*1e-8
            err = 0
            j = 0
            for start in range(0, len(x_train), batch_size):
                j += 1
                end = start + batch_size
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]

                
                output = self.forward_propagation(x_batch)

                err += np.sum(self.loss(y_batch, output))


                if val_set is None:
                    if ((i+1) % verbose == 0) or (i == 0) or (i == epochs-1):
                        print(f"{j-1}/{group} [{'x'*(j*20//group)}{'-'*(20 - j*20//group)}] Epoch {i+1}/{epochs}, Error: {'%.4f' % (err/j)}", end= "\r")
                else:
                    val_out = self.forward(val_set[0])
                    val_err = np.sum(self.loss(val_set[1], val_out))
                    if ((i+1) % verbose == 0) or (i == 0) or (i == epochs-1):
                        print(f"{j-1}/{group} [{'x'*(j*20//group)}{'-'*(20 - j*20//group)}] -> Epoch {i+1}/{epochs}, Error: {'%.4f' % (err/j)}, Val_Error: {'%.4f' % val_err}", end= "\r")
                error = self.loss_derivative(y_batch, output)
                # print(error)
                self.backward_propagation(error, learning_rate)
            if ((i+1) % verbose == 0) or (i == 0) or (i == epochs-1): print()
            err /= j
            self.errors.append(err)
            if val_set is not None: 
                self.val_errors.append(val_err)
            # if i % 5 == 4: 
            # print(f"Epoch {i+1}/{epochs}, Error: {err}, ",end="")
            # if val_set is not None:            # and i % 5 == 4: 
            #     print(f"val_err: {val_err}") 


        for i,l in enumerate(self.layers):
            print(f"layer {i}\n{l.weights}")


    def summary(self):
        print(f"Number of layers: {len(self.layers)}")
        print("------------------------")
        for i,layer in enumerate(self.layers):
            print(f"Layer {i}\n\tNumber of neurons: {layer.size}\n\tActivation: {layer.act_str}")


    def predict(self, X_test):
        y_pred = self.forward(X_test)
        return y_pred


    # TODO
    # parametrik yap lossları
    # @staticmethod
    # def loss(y_true, y_pred):
    #     y_pred = y_pred.reshape(-1)

    #     y_pred[y_pred == 1] = 1 - 1e-10
    #     y_pred[y_pred == 0] = 1e-10
        
    #     # print("y_true shape", y_true.shape)
    #     # print("y_pred shape", y_pred.shape)
    #     """Binary cross-entropy loss."""
    #     los = 0
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings("error")
    #         try:
    #             los = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    #         except Warning as e:
    #             print(e)
    #             print("y_pred", y_pred)
    #             sys.exit()
    #     return los


    # @staticmethod
    # def loss_derivative(y_true, y_pred):
    #     """Derivative of binary cross-entropy loss."""
    #     # print("y_true shape", y_true.shape)
    #     # print("y_pred shape", y_pred.shape)
    #     y_pred = y_pred.reshape(-1)
    #     # print("AAy_true shape", y_true.shape)
    #     # print("AAy_pred shape", y_pred.shape)
    #     # y_pred[y_pred == 1] = 1 - 1e-10
    #     # y_pred[y_pred == 0] = 1e-10
    #     return -(y_true / y_pred) - (1 - y_true) / (1 - y_pred)


    # @staticmethod
    # ##MSE
    # def loss(y_true, y_pred):
    #     y_pred = y_pred.reshape(-1)
    #     mse = np.mean((y_true - y_pred) ** 2)
    #     return mse
    
    # @staticmethod
    # def loss_derivative(y_true, y_pred):
    #     y_pred = y_pred.reshape(-1)
    #     derivative = -2 * (y_true - y_pred) / y_true.size
    #     return derivative