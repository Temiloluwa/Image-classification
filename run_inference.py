import os
import torch
import numpy as np
import onnx
import argparse
from torchvision import models
from data import preprocess_data
from utils import read_imagenet_classnames

parser = argparse.ArgumentParser(description='Inference Trained Model')
parser.add_argument('data', metavar='DIR', help='default data path')
parser.add_argument('-bs', '--batch-size', metavar='BS', default=2, help='maximum batchsize')
parser.add_argument('-tp', '--top-predictions', metavar='NUMPRED',\
                     default=5, help='number of top predictions per sample')
parser.add_argument('-mt', '--model-type', metavar='MODELTYPE', choices=['pytorch', 'onnx'],\
                    default='pytorch', help='onnx or pytorch')


def run_inference(model, data, onnx = False):
    input_data, _ = data
    predictions = model(input_data)
    if not onnx:
        probabilities = torch.softmax(predictions, 1)
        pred_indices = torch.argsort(probabilities, 1,\
                            descending=True)[:, : args.top_predictions]
        probabilities = torch.gather(probabilities, 1, pred_indices)
        probabilities = (probabilities * 100).detach().numpy()
        pred_indices = pred_indices.detach().numpy()

    return probabilities, pred_indices


def display_results(data, predictions):
    input_data, file_names = data
    probabilities, pred_indices = predictions

    assert len(input_data) == len(file_names)  \
            == len(probabilities) == len(pred_indices),\
            "check batch size of data and prediction"

    for i, file in enumerate(file_names):
        prob, idx = probabilities[i], pred_indices[i]
        prediction = [f"{imagenet_classes[idx[j]][0]} with probability {prob[j]:0.2f}%" \
                        for j in range(len(prob))]
        prediction = "\n".join(prediction)
        print(f"\n file {file} prediction: {prediction}")

    
def export_to_onnx(model):
    sample_input = torch.zeros((1, 3, 224, 224))
    onnx.export(model, sample_input, 'checkpoints/model.onnx')


def prepare_model():
    if args.model_type == 'pytorch':
        model = models.resnet18(pretrained=True)
    return model
    

if __name__ == "__main__":
    args = parser.parse_args()
    imagenet_classes = read_imagenet_classnames(\
        os.path.join(args.data, "imagenet_classnames.txt"))
    model = prepare_model()
    data = preprocess_data(args.data)
    predictions = run_inference(model, data)
    display_results(data, predictions)
    