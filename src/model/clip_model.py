from model.base_model import BaseModel
from transformers import CLIPModel as TCLIPModel, CLIPImageProcessor, CLIPTokenizer

class CLIPModel(BaseModel):
    def __init__(self, gpu, model="openai/clip-vit-base-patch32", snapshot=None):
        super().__init__(img_size=224, gpu=gpu)

        self.model = TCLIPModel.from_pretrained(snapshot if snapshot is not None else model).to(gpu)
        self.image_processor = CLIPImageProcessor.from_pretrained(model)
        self.tokenizer = CLIPTokenizer.from_pretrained(model)

    def preprocess_images(self, images):
        return self.image_processor(images, return_tensors="pt", do_resize=False, do_center_crop=False).to(self.gpu)

    def preprocess_texts(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True).to(self.gpu)

    def score(self, images, texts):
        image_inputs = self.preprocess_images(images)
        text_inputs = self.preprocess_texts(texts)
        
        result = self.model(**image_inputs, **text_inputs)
        return result["logits_per_image"]
    
    def get_image_features(self, images):
        image_inputs = self.preprocess_images(images)
        return self.model.get_image_features(**image_inputs)
    
    def get_text_features(self, texts):
        text_inputs = self.preprocess_texts(texts)
        return self.model.get_text_features(**text_inputs)
        