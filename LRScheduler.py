class LRScheduler:
    class none:
        def calc(self, learning_rate, epoch):
            return learning_rate
        
    class TimeBasedDecay:
        def __init__(self, decay = 0.01):
            self.decay = decay

        def calc(self, learning_rate, epoch):
            return learning_rate / (1 + self.decay * epoch)
        
    class StepDecay:
        def __init__(self, drop_rate = 0.1, epochs_drop = 20):
            self.drop_rate = drop_rate
            self.epochs_drop = epochs_drop

        def calc(self, learning_rate, epoch):
            return learning_rate * (self.drop_rate ** (epoch // self.epochs_drop))
        
    class ExponentialDecay:
        def __init__(self, decay_rate = 0.9, decay_steps = 1000):
            self.decay_rate = decay_rate
            self.decay_steps = decay_steps

        def calc(self, learning_rate, epoch):
            return learning_rate * (self.decay_rate ** (epoch / self.decay_steps))
            
