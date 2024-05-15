from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, img_size, gpu):
        self.img_size = img_size
        self.gpu = gpu
    
    @abstractmethod
    def preprocess_images(self, images):
        pass

    @abstractmethod
    def preprocess_texts(self, texts):
        pass

    @abstractmethod
    def score(self, images, texts):
        pass

    @abstractmethod
    def get_image_features(self, images):
        pass
    
    @abstractmethod
    def get_text_features(self, texts):
        pass