import numpy as np
class Optimizers:
    class SGD:
        def __init__(self):
            pass
        def update(self, w, dw, b, db, learning_rate):
            print("DW", dw.shape)
            print("DB", db.shape)
            w -= learning_rate * dw
            b -= learning_rate * db
            return w, b
        
    class Momentum:
        def __init__(self, beta = 0.9):
            self.beta = beta
            self.v_w = None
            self.v_b = None

        def update(self, w, dw, b, db, learning_rate):
            if self.v_w is None and self.v_b is None:
                self.v_w = np.zeros_like(dw)
                self.v_b = np.zeros_like(db)
            print(self.v_w.shape)
            print("dw", dw.shape)
            self.v_w = self.beta * self.v_w - learning_rate * dw
            self.v_b = self.beta * self.v_b - learning_rate * db
            w = w + self.v_w
            b = b + self.v_b
            return w, b
    class Adam:
        def __init__(self, beta1 = 0.9,
                     beta2 = 0.999, epsilon = 1e-8):
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m_w, self.v_w = np.zeros(), np.zeros()
            self.m_b, self.v_b = np.zeros(), np.zeros()
            self.t = 0

        def update(self, w, dw, b, db, learning_rate):
            self.t += 1
            self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
            self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)
            m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w / (1 - self.beta2 ** self.t)

            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)
            m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

            w -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            return w, b