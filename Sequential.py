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
    """
    A class that represents a sequential neural network model.

    This model is composed of a stack of layers added in a sequential manner. It supports various optimizers,
    learning rate schedulers, and regularization techniques such as L1 and L2 regularization.

    Attributes
    ----------
    layers : list
        A list that holds all the layers within the model.
    errors : list
        A list that records the training error after each epoch.
    val_errors : list or None
        A list that records the validation error after each epoch, if a validation set is provided.
    optimizer : Optimizers class instance
        The optimization algorithm used for training the model.
    scheduler : LRScheduler class instance
        The learning rate scheduling strategy.
    stop : bool
        A flag used to stop training early if the validation error does not decrease.
    metrics : Metrics class instance or None
        The metrics used to evaluate the model performance.
    l1_lambda : float
        The regularization strength for L1 regularization.
    l2_lambda : float
        The regularization strength for L2 regularization.
    val_best : Sequential instance or None
        A deep copy of the model that achieved the best validation performance.
    val_best_loss : float
        The best validation loss achieved during training.

    Methods
    -------
    add(layer):
        Adds a layer to the model.
    build(input_shape):
        Initializes the model's layers based on the input shape.
    forward_propagation(input_data):
        Performs a forward pass through the model.
    forward(input_data):
        An alias for forward_propagation, performs a forward pass through the model.
    backward_propagation(output_error, learning_rate):
        Performs a backward pass through the model, updating the parameters.
    train(x_train, y_train, epochs, learning_rate, batch_size, val_set, loss, verbose):
        Trains the model on the provided training data.
    summary():
        Prints a summary of the model's architecture and parameters.
    predict(X_test):
        Predicts the outputs for the given test data.
    """
    def __init__(self, optimizer = Optimizers.SGD(), scheduler = LRScheduler.none(),
                 stop = False, metrics_list=None, l1_lambda = 0.0, l2_lambda = 0.0):
        """
        Initializes the Sequential model with the specified optimizer, learning rate scheduler,
        early stopping criterion, metrics, and regularization parameters.

        Parameters
        ----------
        optimizer : Optimizers class instance, optional
            The optimization algorithm to use (default is SGD).
        scheduler : LRScheduler class instance, optional
            The learning rate scheduling strategy to use (default is none).
        stop : bool, optional
            Whether to stop training early if the validation performance does not improve (default is False).
        metrics_list : list of str, optional
            The list of metric names to evaluate the model performance (default is None).
        l1_lambda : float, optional
            The L1 regularization strength (default is 0.0).
        l2_lambda : float, optional
            The L2 regularization strength (default is 0.0).
        """
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
        """
        Trains the neural network model on the provided training data.

        Parameters
        ----------
        x_train : np.ndarray
            The input features of the training data.
        y_train : np.ndarray
            The target values (labels) of the training data.
        epochs : int
            The number of times to iterate over the entire dataset.
        learning_rate : float
            The initial learning rate for the optimizer.
        batch_size : int, optional
            The number of samples per batch to load (default is 32).
        val_set : tuple of np.ndarray, optional
            A tuple (x_val, y_val) containing the input features and target values (labels) of the validation data (default is None).
        loss : str, optional
            The name of the loss function to use (default is "MSE").
        verbose : int, optional
            The verbosity mode, specifies how often to print out training progress (default is 5). Printing happens at every epoch if verbose is 1, every 5 epochs if verbose is 5, and so on.

        Overview
        --------
        This method orchestrates the training process of the model for a specified number of epochs or until early stopping criteria are met if a validation set and stopping flag are provided. The training process involves:
        
        - Initializing the model's layers and weights.
        - Iterating over the training data in batches.
        - Performing forward propagation to calculate predictions.
        - Calculating the loss between predictions and true values.
        - Performing backward propagation to update weights and biases.
        - Optionally evaluating the model on a validation set to monitor performance.
        - Applying learning rate scheduling if specified.
        - Implementing early stopping based on validation loss improvement.

        The function keeps track of training and validation errors, and it can optionally print out progress information depending on the verbosity level set by the user.

        Note
        ----
        The method automatically handles the detection of problem types (binary, multiclass, or multilabel classification) based on the dimensions and values of `y_train` and the activation function of the last layer of the model.

        Returns
        -------
        The best model based on validation loss if early stopping is enabled and a validation set is provided; otherwise, the function returns None.
        """
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
                        self.val_best_loss = val_err
                        self.val_best = copy.deepcopy(self)
                        early = 0
                    else:
                        early += 1
                elif self.stop:
                    return self.val_best
                

    def summary(self):
        print(f"Optimizer: {str(self.optimizer)}")
        print(f"Number of layers: {len(self.layers)}")
        print("------------------------")
        for i,layer in enumerate(self.layers):
            print(f"Layer {i}\n\tNumber of neurons: {layer.size}\n\tActivation: {layer.act_str}")

        


    def predict(self, X_test):
        y_pred = self.forward(X_test)
        return y_pred