from model.base_model import BaseModel
from viper_gpt.xvlm import XVLMBase
from transformers import BertTokenizer
import torch
from torchvision import transforms
from PIL import Image
import re

class XVLMModel(BaseModel):
    def __init__(self, gpu):
        super().__init__(img_size=384, gpu=gpu)

        self.max_words = 30
        config_xvlm = {
            'image_res': self.img_size,
            'patch_size': 32,
            'text_encoder': 'bert-base-uncased',
            'block_num': 9,
            'max_tokens': 40,
            'embed_dim': 256,
        }

        vision_config = {
            'vision_width': 1024,
            'image_res': 384,
            'window_size': 12,
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32]
        }

        model = XVLMBase(config_xvlm, use_contrastive_loss=True, vision_config=vision_config)
        checkpoint = torch.load('../data/models/xvlm_vipergpt/retrieval_mscoco_checkpoint_9.pth', map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        model.load_state_dict(state_dict, strict=False)

        self.model = model.to(gpu)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.transform = transforms.Compose([
            transforms.Lambda(lambda t: t * (1/255)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @staticmethod
    def __pre_caption__(caption, max_words):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        if not len(caption):
            raise ValueError("pre_caption yields invalid text")

        return caption

    def preprocess_images(self, images):
        images = [self.transform(image) for image in images]
        images = torch.stack(images, dim=0).to(self.gpu)
        image_embeds, _ = self.model.get_vision_embeds(images)
        return image_embeds

    def preprocess_texts(self, texts):
        texts = [self.__pre_caption__(text, self.max_words) for text in texts]
        text_input = self.tokenizer(texts, padding='longest', return_tensors="pt").to(self.gpu)
        text_ids, text_atts = text_input.input_ids, text_input.attention_mask
        text_embeds = self.model.get_text_embeds(text_ids, text_atts)
        return text_embeds

    def score(self, images, texts):
        text_inputs = self.preprocess_texts(texts)
        image_inputs = self.preprocess_images(images)

        image_feat, text_feat = self.model.get_features(image_inputs, text_inputs)
        return image_feat @ text_feat.t()
    
    def get_image_features(self, images):
        image_inputs = self.preprocess_images(images)
        return self.model.get_features(image_embeds=image_inputs)
    
    def get_text_features(self, texts):
        text_inputs = self.preprocess_texts(texts)
        return self.model.get_features(text_embeds=text_inputs)