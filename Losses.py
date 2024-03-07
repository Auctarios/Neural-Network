import numpy as np
class Losses:
    def __init__(self, loss: str = "MSE"):
        avail_types = ["MSE", "BinaryCrossEntropy", "MeanAbsoluteError"]
        
        if loss not in avail_types:
            raise ValueError(f"Given loss function is not availabile, available loss functions are: {avail_types}")
        
        self.type = loss

        self.loss = getattr(self, loss)
        loss_deri = loss + "_deri"
        self.loss_derivative = getattr(self, loss_deri)

    @staticmethod
    def MSE(y_true, y_pred):
        # print("def y_true shape", y_true.shape)
        # print("def y_pred shape", y_pred.shape)
        y_pred = y_pred.reshape(-1)
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    @staticmethod
    def MSE_deri(y_true, y_pred):
        # print("deri y_true shape", y_true.shape)
        # print("deri y_pred shape", y_pred.shape)
        y_pred = y_pred.reshape(-1)
        derivative = -2 * (y_true - y_pred) / y_true.size
        return derivative
    
    @staticmethod
    def BinaryCrossEntropy(y_true, y_pred):
# a[a == 0] = 1e-10
# a[a == 1] = 1 - 1e-10
        y_pred = y_pred.reshape(-1)
        y_pred[y_pred == 0] = 1e-10
        y_pred[y_pred == 1] = 1 - (1e-10)
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return bce
    #squeze unqueze
    
    @staticmethod
    def BinaryCrossEntropy_deri(y_true, y_pred):
        y_pred = y_pred.reshape(-1)
        bce = (-1/y_true.size) * ((y_true/y_pred) + ((y_true - 1) / (1 - y_pred)))
        return bce
    
    @staticmethod
    def MeanAbsoluteError(y_true, y_pred):
        y_pred = y_pred.reshape(-1)
        return np.mean(abs(y_true - y_pred))

    @staticmethod
    def MeanAbsoluteError_deri(y_true, y_pred):
        y_pred = y_pred.reshape(-1)
        y_pred[y_pred > y_true] = 1
        y_pred[y_pred < y_true] = -1
        y_pred[y_pred == y_true] = 0
        return y_pred