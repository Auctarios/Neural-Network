import numpy as np
class Optimizers:
    class SGD:
        def __init__(self):
            pass
        def update(self, w, dw, b, db, learning_rate):
            w -= learning_rate * dw
            b -= learning_rate * db
            return w, b
        
        def get_params(self):
            return ([], {})
        
        def __str__(self):
            return "SGD"
        
    class AdaGrad:
        def __init__(self, eps = 1e-7):
            self.eps = eps
            self.r_w = 0
            self.r_b = 0

        def get_params(self):
            return ([self.eps], {})
        
        def update(self, w, dw, b, db, learning_rate):
            self.r_w += dw * dw
            self.r_b += db * db
            w += (-1.0) * (learning_rate / (self.eps + np.sqrt(self.r_w))) * dw
            b += (-1.0) * (learning_rate / (self.eps + np.sqrt(self.r_b))) * db
            return w, b
        
        def __str__(self):
            return "AdaGrad"
        
    class Momentum:
        def __init__(self, beta = 0.9):
            self.beta = beta
            self.v_w = None
            self.v_b = None

        def get_params(self):
            return ([self.beta], {})

        def update(self, w, dw, b, db, learning_rate):
            if self.v_w is None and self.v_b is None:
                self.v_w = np.zeros_like(dw)
                self.v_b = np.zeros_like(db)
            self.v_w = self.beta * self.v_w - learning_rate * dw
            self.v_b = self.beta * self.v_b - learning_rate * db
            w = w + self.v_w
            b = b + self.v_b
            return w, b
        
        def __str__(self):
            return "Momentum"
    
    class RMSProp:
        def __init__(self, eps = 1e-8, alpha = 0.99):
            self.eps = eps
            self.r_w = 0
            self.r_b = 0
            self.alpha = alpha

        def update(self, w, dw, b, db, learning_rate):
            self.r_w = self.alpha * self.r_w + (1 - self.alpha) * (dw * dw)
            self.r_b = self.alpha * self.r_b + (1 - self.alpha) * (db * db)
            w += (-1) * (learning_rate/np.sqrt(self.eps + self.r_w)) * dw
            b += (-1) * (learning_rate/np.sqrt(self.eps + self.r_b)) * db
            return w, b
        
        def get_params(self):
            return ([self.eps, self.alpha], {})
        
        def __str__(self):
            return "RMSProp"


    class Adam:
        def __init__(self, beta1 = 0.9,
                     beta2 = 0.999, epsilon = 1e-8):
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.s_w = None
            self.s_b = None
            self.r_w = None
            self.r_b = None
            self.t = None

        def get_params(self):
            return ([self.beta1, self.beta2, self.epsilon], {})

        def update(self, w, dw, b, db, learning_rate):
            if self.s_w is None:
                self.s_w = np.zeros_like(dw)
                self.s_b= np.zeros_like(db)
                self.r_w = np.zeros_like(dw)
                self.r_b = np.zeros_like(db)
                self.t = 0
            
            self.t += 1

            self.s_w = (self.beta1 * self.s_w) + ((1 - self.beta1) * dw)
            self.s_b = (self.beta1 * self.s_b) + ((1 - self.beta1) * db)

            self.r_w = (self.beta2 * self.r_w) + ((1 - self.beta2) * dw * dw)
            self.r_b = (self.beta2 * self.r_b) + ((1 - self.beta2) * db * db)
            w += (-1 * learning_rate) * (((self.s_w) / (1 - (self.beta1 ** self.t))) / 
                                     (self.epsilon + np.sqrt((self.r_w/(1 - (self.beta2 ** self.t))))))
            b += (-1 * learning_rate) * (((self.s_b) / (1 - (self.beta1 ** self.t))) / 
                                     (self.epsilon + np.sqrt((self.r_b/(1 - (self.beta2 ** self.t))))))
            return w, b
        
        def __str__(self):
            return "Adam"