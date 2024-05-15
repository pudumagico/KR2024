from model.base_model import BaseModel
from transformers import BlipModel, BlipImageProcessor, BertTokenizerFast, BatchEncoding

class BLIPModel(BaseModel):
    def __init__(self, gpu):
        super().__init__(img_size=384, gpu=gpu)

        self.model = BlipModel.from_pretrained("Salesforce/blip-itm-base-coco").to(gpu)
        self.image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.tokenizer = BertTokenizerFast.from_pretrained("Salesforce/blip-itm-base-coco")

    def preprocess_images(self, images):
        return self.image_processor(images, return_tensors="pt", do_resize=False).to(self.gpu)

    def preprocess_texts(self, texts):
        text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        return BatchEncoding({k: text_inputs[k] for k in ("input_ids", "attention_mask")}).to(self.gpu)

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