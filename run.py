import os
import torch
import numpy as np
from torchvision import models
from data import preprocess_data, DEFAULT_DATA_PATH
from utils import read_imagenet_classnames

TOP_PREDICTIONS = 5

def run_inference(model, input_data):
    predictions = model(input_data)
    probabilities = torch.softmax(predictions, 1)
    pred_indices = torch.argsort(probabilities, 1,\
                         descending=True)[:, : TOP_PREDICTIONS]
    probabilities = torch.gather(probabilities, 1, pred_indices)
    probabilities = (probabilities * 100).detach().numpy()
    pred_indices = pred_indices.detach().numpy()
    return probabilities, pred_indices


def prepare_model():
    model = models.resnet18(pretrained=True)
    return model
    

if __name__ == "__main__":
    imagenet_classes = read_imagenet_classnames(os.path.join(DEFAULT_DATA_PATH, "imagenet_classnames.txt"))
    model = prepare_model()
    input_data, file_names = preprocess_data()
    probabilities, pred_indices = run_inference(model, input_data)
    for i, file in enumerate(file_names):
        prob, idx = probabilities[i], pred_indices[i]
        prediction = [f"{imagenet_classes[idx[j]][0]} with probability {prob[j]:0.2f}%" for j in range(len(prob))]
        prediction = "\n".join(prediction)
        print(f"\n file {file} prediction: {prediction}")
    