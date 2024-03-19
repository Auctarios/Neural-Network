import numpy as np
class Losses:
    
    """
    A class that encapsulates various loss functions for neural network training.

    Attributes
    ----------
    type : str
        The type of loss function to use. Supported loss functions are 'MSE' (Mean Squared Error), 'BinaryCrossEntropy', 
        'MeanAbsoluteError', and 'CrossEntropy'.
    loss : function
        The loss function corresponding to `type`.
    loss_derivative : function
        The derivative of the loss function corresponding to `type`.
    """
    def __init__(self, loss: str = "MSE"):
        """
        Initializes the Losses object with the specified loss function type.

        Parameters
        ----------
        loss : str, optional
            The type of loss function to use. Default is 'MSE'.

        Raises
        ------
        ValueError
            If the specified loss function type is not supported.
        """
        avail_types = ["MSE", "BinaryCrossEntropy", "MeanAbsoluteError", "CrossEntropy"]
        
        if loss not in avail_types:
            raise ValueError(f"Given loss function is not availabile, available loss functions are: {avail_types}")
        
        self.type = loss

        self.loss = getattr(self, loss)
        loss_deri = loss + "_deri"
        self.loss_derivative = getattr(self, loss_deri)

    @staticmethod
    def MSE(y_true, y_pred):
        """
        Calculates the Mean Squared Error between the true labels and predictions.
        """
        y_pred = y_pred.reshape(-1)
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    @staticmethod
    def MSE_deri(y_true, y_pred):
        """
        Calculates the derivative of the Mean Squared Error loss function.
        """
        y_pred = y_pred.reshape(-1)
        derivative = -2 * (y_true - y_pred) / y_true.size
        return derivative
    
    @staticmethod
    def BinaryCrossEntropy(y_true, y_pred):
        """
        Calculates the Binary Cross-Entropy loss between the true labels and predictions.
        """
        y_pred = y_pred.reshape(-1)
        y_pred[y_pred == 0] = 1e-10
        y_pred[y_pred == 1] = 1 - (1e-10)
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return bce
    
    @staticmethod
    def BinaryCrossEntropy_deri(y_true, y_pred):
        """
        Calculates the derivative of the Binary Cross-Entropy loss function.
        """
        y_pred = y_pred.reshape(-1)
        bce = (-1/y_true.size) * ((y_true/y_pred) + ((y_true - 1) / (1 - y_pred)))
        return bce
    
    @staticmethod
    def MeanAbsoluteError(y_true, y_pred):
        """
        Calculates the Mean Absolute Error between the true labels and predictions.
        """
        y_pred = y_pred.reshape(-1)
        return np.mean(abs(y_true - y_pred))

    @staticmethod
    def MeanAbsoluteError_deri(y_true, y_pred):
        """
        Calculates the derivative of the Mean Absolute Error loss function.
        """
        y_pred = y_pred.reshape(-1)
        y_pred[y_pred > y_true] = 1
        y_pred[y_pred < y_true] = -1
        y_pred[y_pred == y_true] = 0
        return y_pred
    
    @staticmethod
    def CrossEntropy(y_true, y_pred):
        """
        Calculates the Cross-Entropy loss between the true labels and the predictions for multi-class classification.
        """
        m = y_pred.shape[0]

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_log_probs = -np.log(y_pred_clipped[range(m), y_true.argmax(axis=1)])
        data_loss = np.sum(correct_log_probs) / m
        return data_loss
    
    @staticmethod
    def CrossEntropy_deri(y_true, y_pred):
        """
        Calculates the derivative of the Cross-Entropy loss function.
        """
        m = y_pred.shape[0]
        grad = y_pred - y_true
        grad /= m
        return grad