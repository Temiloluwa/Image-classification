import os
import torch
from PIL import Image
from torchvision import transforms, datasets


DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
MAX_BATCHSIZE = 2

transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    
])

def preprocess_image(path: str):
    """ Preprocess image """
    with Image.open(path) as im:
        im = transformations(im)
    return im

def preprocess_data(path: str = DEFAULT_DATA_PATH):
    """ Batch files in data folder """
    data_tensors = []
    file_names = []
    for file_path in os.listdir(path):
        file_name, extension = os.path.splitext(file_path)
        if extension not in [".jpg", ".png"]:
            continue
        file_path = os.path.join(path, file_path)
        data_tensors.append(preprocess_image(file_path))
        file_names.append(file_name)

    data_tensors = torch.stack(data_tensors)
    return data_tensors, file_names
