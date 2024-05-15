from abc import ABC, abstractmethod

class BaseObjectDetector(ABC):
    def __init__(self, gpu):
        self.gpu = gpu
    
    @abstractmethod
    def detect_objects(self, image, classes, threshold, k):
        pass