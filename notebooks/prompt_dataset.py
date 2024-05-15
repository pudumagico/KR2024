import torch 
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import crop, resize, pad
from math import tanh
import pandas as pd

def scaling(x, ceiling=3):
    return (1 - tanh(x * 2)) * ceiling

def get_scaled_bbox(entry, img_height, img_width, padding_scale_ceiling=1):
    padding_w = scaling(entry["bbox_w"] / img_width, padding_scale_ceiling) * entry["bbox_w"]
    padding_h = scaling(entry["bbox_h"] / img_height, padding_scale_ceiling) * entry["bbox_h"]

    return (
        int(max(entry["bbox_y"] - padding_h, 0)),
        int(max(entry["bbox_x"] - padding_w, 0)),
        int(min(entry["bbox_h"]+2*padding_h, img_height-max(entry["bbox_y"] - padding_h, 0))),
        int(min(entry["bbox_w"]+2*padding_w, img_width-max(entry["bbox_x"] - padding_w, 0)))
    )

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, prompt_transform, img_size=224, mode="pad"):
        self.df = df 
        self.prompt_transform = prompt_transform
        self.img_size = img_size
        self.mode = mode

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        image = read_image(f"../data/images/{entry['image_id']}.jpg", ImageReadMode.RGB)

        # crop bounding box
        y,x,h,w = get_scaled_bbox(entry, image.shape[1], image.shape[2])
        image = crop(image, y, x, h, w)

        if self.mode == "pad":
            # resize and scale (maintain aspect ratio)
            if entry["bbox_h"] > entry["bbox_w"]:
                resize_dimensions = (self.img_size, 2*round((self.img_size*entry["bbox_w"]/entry["bbox_h"])/2)) 
            else:
                resize_dimensions = (2*round((self.img_size*entry["bbox_h"]/entry["bbox_w"])/2), self.img_size)
            image = resize(image, resize_dimensions, antialias=True)

            # pad the image to square dimensions
            image = pad(image, ((self.img_size - resize_dimensions[1])//2, (self.img_size - resize_dimensions[0])//2))

        elif self.mode == "scale":
            # resize and scale the image to the target dimensions
            image = resize(image, (self.img_size, self.img_size), antialias=True)

        else: 
            raise RuntimeError("Unsupported image processing mode!")

        return (image, self.prompt_transform(entry), entry['y'])