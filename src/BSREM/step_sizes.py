from abc import ABC, abstractmethod

class StepSize():

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, algorithm):
        pass

class FixedStepSize(StepSize):

    def __init__(self, step_size):
        self.step_size = step_size

    def __call__(self, algorithm):
        return self.step_size
    
class HarmonicDecayStepSize(StepSize):

    def __init__(self, step_size, decay):
        self.step_size = step_size
        self.decay = decay

    def __call__(self, algorithm):
        return self.step_size / (1 + algorithm.iteration * self.decay)