import os
import argparse
import torch
from data import preprocess_data
from utils import read_imagenet_classnames, display_results, run_inference, parse_base64
from torchvision import models

parser = argparse.ArgumentParser(description='Inference Trained Model')
parser.add_argument('--data', metavar='DIR', default='./data', help='default data path')
parser.add_argument('-bs', '--batch-size', metavar='BS', default=2, help='maximum batchsize')
parser.add_argument('-tp', '--top-predictions', metavar='NUMPRED',\
                     default=5, help='number of top predictions per sample')
parser.add_argument('-exp', '--export', action="store_true",help='export model to onnx')


def export_model(model):
    model_path = 'checkpoints/model.pt'
    sample_input = torch.randn((1, 3, 256, 256))
    model = model.cpu()
    model.eval()
    model = torch.jit.trace(model, sample_input)
    torch.jit.save(model, model_path)    


if __name__ == "__main__":
    args = parser.parse_args()
    model = models.resnet18(pretrained=True)
    if args.export:
        export_model(model)
    else:
        imagenet_classes = read_imagenet_classnames("cache/imagenet_classnames.txt")
        data = preprocess_data("cache")
        predictions = run_inference(model, data[0], args.top_predictions)
        display_results(data, predictions, imagenet_classes)
