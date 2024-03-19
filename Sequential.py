import sys
from Layer import *
import warnings
import numpy as np
from Losses import Losses
from Optimizers import Optimizers
from LRScheduler import LRScheduler
from Metrics import Metrics
import copy
class Sequential:
    def __init__(self, optimizer = Optimizers.SGD(), scheduler = LRScheduler.none(),
                 stop = False, metrics_list=None, l1_lambda = 0.0, l2_lambda = 0.0):
        self.layers = []
        self.errors = []
        self.val_errors = None
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stop = stop
        self.metrics = None
        self.training_metrics = None
        self.validation_metrics = None
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.val_best = None
        self.val_best_loss = float('inf')
        if metrics_list is not None:
            self.metrics = Metrics(metrics_list)
            self.training_metrics = []



    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError(f"Given layer is {type(layer)}, instead it should be Layer")
        
        self.layers.append(layer)

    def build(self, input_shape):
        prev_output_size = input_shape
        for layer in self.layers:
            if layer.weights is None:
                args, kwargs = self.optimizer.get_params()
                layer.initialize(prev_output_size, type(self.optimizer)(*args, **kwargs))
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
            if self.l1_lambda > 0:
                layer.weights -= learning_rate * self.l1_lambda * np.sign(layer.weights)

            if self.l2_lambda > 0:
                layer.weights -= learning_rate * self.l2_lambda * layer.weights

    def train(self, x_train, y_train, epochs, learning_rate, batch_size = 32, val_set: tuple = None, loss: str = "MSE", verbose: int = 5):
        #binary, multiclass, multilabel
        if y_train.ndim != 1:
            problem_type = "multiclass" if self.layers[-1].act_str == "softmax" else "multilabel"
        elif y_train.ndim == 1:
            problem_type = "binary"
        else:
            raise ValueError("Something broke")
        
        
        
        
        
        if self.val_errors is None and val_set is not None:
            self.val_errors = []
            self.validation_metrics = []
        looss = Losses(loss)
        self.loss = looss.loss
        lr = learning_rate
        self.loss_derivative = looss.loss_derivative
        print(x_train.shape)
        self.build(x_train.shape[1])
        if val_set is not None:
            val_err = -1
        group = len(x_train)//batch_size

        
        early = 0
        for i in range(epochs):
            if len(self.errors) != 0:
                lr = self.scheduler.calc(learning_rate, len(self.errors))
            err = 0
            j = 0
            for start in range(0, len(x_train), batch_size):
                j += 1
                end = start + batch_size
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]

                
                output = self.forward_propagation(x_batch)

                #TODO
                #output.shape (number of neurons in the last layer) ?= y_batch.shape

                err += np.sum(self.loss(y_batch, output))


                if val_set is None:
                    if self.metrics:
                        train_str, train_metrics = self.metrics.calc(output, y_batch, problem_type=problem_type)
                        self.training_metrics.append(train_metrics)
                    if ((i+1) % verbose == 0) or (i == 0) or (i == epochs-1):
                        if self.metrics:
                            print(f"{j-1}/{group} [{'x'*(j*20//group)}{'-'*(20 - j*20//group)}] Epoch {i+1}/{epochs}, Error: {'%.4f' % (err/j)}, LR: {'%.8f' % lr}, Train: {train_str}", end= "\r")
                        else:
                            print(f"{j-1}/{group} [{'x'*(j*20//group)}{'-'*(20 - j*20//group)}] Epoch {i+1}/{epochs}, Error: {'%.4f' % (err/j)}, LR: {'%.8f' % lr}", end= "\r")

                else:
                    val_out = self.forward(val_set[0])
                    val_err = np.sum(self.loss(val_set[1], val_out))
                    if self.metrics:
                        train_str, train_metrics = self.metrics.calc(output, y_batch, problem_type=problem_type)
                        val_str, val_metrics = self.metrics.calc(val_out, val_set[1], problem_type=problem_type)
                        self.training_metrics.append(train_metrics)
                        self.validation_metrics.append(val_metrics)
                    if (((i+1) % verbose == 0) or (i == 0) or (i == epochs-1)):
                        if self.metrics:
                            print(f"{j-1}/{group} [{'x'*(j*20//group)}{'-'*(20 - j*20//group)}] -> Epoch {i+1}/{epochs}, Error: {'%.4f' % (err/j)}, Val_Error: {'%.4f' % val_err}, LR: {'%.8f' % lr}, Train: {train_str}, Val: {val_str}", end= "\r")
                        else:
                            print(f"{j-1}/{group} [{'x'*(j*20//group)}{'-'*(20 - j*20//group)}] -> Epoch {i+1}/{epochs}, Error: {'%.4f' % (err/j)}, Val_Error: {'%.4f' % val_err}, LR: {'%.8f' % lr}", end= "\r")
                error = self.loss_derivative(y_batch, output)
                self.backward_propagation(error, lr)
            if ((i+1) % verbose == 0) or (i == 0) or (i == epochs-1): print()
            err /= j
            self.errors.append(err)
            if val_set is not None: 
                self.val_errors.append(val_err)
                if early < epochs / 10:
                    if val_err < self.val_best_loss:
                        self.val_best = None
                        self.val_best = copy.deepcopy(self)
                        early = 0
                        self.val_best_loss = val_err
                    else:
                        early += 1
                elif self.stop:
                    return self.val_best
                
            # if i % 5 == 4: 
            # print(f"Epoch {i+1}/{epochs}, Error: {err}, ",end="")
            # if val_set is not None:            # and i % 5 == 4: 
            #     print(f"val_err: {val_err}") 


        # for i,l in enumerate(self.layers):
        #     print(f"layer {i}\n{l.weights}")


    def summary(self):
        print(f"Optimizer: {str(self.optimizer)}")
        print(f"Number of layers: {len(self.layers)}")
        print("------------------------")
        for i,layer in enumerate(self.layers):
            print(f"Layer {i}\n\tNumber of neurons: {layer.size}\n\tActivation: {layer.act_str}")

        


    def predict(self, X_test):
        y_pred = self.forward(X_test)
        return y_pred