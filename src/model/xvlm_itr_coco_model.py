from model.base_model import BaseModel
from x_vlm.models.model_retrieval import XVLM
from x_vlm.models.tokenization_bert import BertTokenizer
from x_vlm.models.tokenization_roberta import RobertaTokenizer
from torchvision import transforms
import ruamel.yaml as yaml
import torch
import torch.nn.functional as F

class XVLMModel(BaseModel):
    def __init__(self, gpu):
        super().__init__(img_size=384, gpu=gpu)

        self.config = yaml.load(open("../externals/x_vlm/configs/config_xvlm_itr_coco.yaml", "r"), Loader=yaml.Loader)

        self.model = XVLM(self.config)
        self.model.load_pretrained("../data/models/xvlm_original_4m/itr_coco/checkpoint_best.pth", self.config, is_eval=True)
        self.model.to(gpu)
        self.model.eval()

        if self.config['use_roberta']:
            self.tokenizer = RobertaTokenizer.from_pretrained(self.config['text_encoder'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.config['text_encoder'])

        self.image_transform = transforms.Compose([
            transforms.Lambda(lambda t: t * (1/255)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def preprocess_images(self, images):
        images = [self.image_transform(image) for image in images]
        images = torch.stack(images, dim=0).to(self.gpu)
        return  self.model.vision_encoder(self.image_transform(images))

    def preprocess_texts(self, texts):
        return self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.config['max_tokens'], return_tensors="pt").to(self.gpu)

    def score(self, images, texts):
        image_embeds = self.get_image_features(images)
        text_embeds = self.get_text_features(texts)
        
        return image_embeds @ text_embeds.t()
    
    def get_image_features(self, images):
        image_inputs = self.preprocess_images(images)
        image_embed = self.model.vision_proj(image_inputs[:, 0, :])
        return F.normalize(image_embed, dim=-1)
    
    def get_text_features(self, texts):
        text_inputs = self.preprocess_texts(texts)
        text_output = self.model.text_encoder(text_inputs.input_ids, attention_mask=text_inputs.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(self.model.text_proj(text_feat[:, 0, :]))
        return text_embed
        