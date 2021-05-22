import os
import argparse
import torch
import onnx
import torch.onnx as torchonnx
from pytorch_data import preprocess_data, run_inference
from utils import read_imagenet_classnames, display_results
from torchvision import models

parser = argparse.ArgumentParser(description='Inference Trained Model')
parser.add_argument('data', metavar='DIR', help='default data path')
parser.add_argument('-bs', '--batch-size', metavar='BS', default=2, help='maximum batchsize')
parser.add_argument('-tp', '--top-predictions', metavar='NUMPRED',\
                     default=5, help='number of top predictions per sample')
parser.add_argument('-exp', '--export', default=False,help='export model to onnx')


def export_to_onnx(model):
    model_path = 'checkpoints/model.onnx'
    sample_input = torch.randn((1, 3, 256, 256))
    torchonnx.export(model, sample_input, model_path)
    model = onnx.load(model_path)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
    onnx.save(model, model_path)


if __name__ == "__main__":
    args = parser.parse_args()
    model = models.resnet18(pretrained=True)
    if args.export:
        export_to_onnx(model)
    else:
        imagenet_classes = read_imagenet_classnames(\
            os.path.join(args.data, "imagenet_classnames.txt"))
        data = preprocess_data(args.data)
        predictions = run_inference(model, data, args.top_predictions)
        display_results(data, predictions, imagenet_classes)
