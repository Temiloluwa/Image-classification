import os
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms


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

def preprocess_data(path: str):
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


def run_inference(model, data, top_predictions):
    input_data, _ = data
    predictions = model(input_data)
    probabilities, pred_indices = F.softmax(predictions, 1).topk(top_predictions)
    probabilities = (probabilities * 100).detach().numpy()
    pred_indices = pred_indices.detach().numpy()
    
    return probabilities, pred_indices

