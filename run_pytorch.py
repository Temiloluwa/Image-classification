import os
import argparse
from data import preprocess_data
from utils import read_imagenet_classnames, export_to_onnx
from torchvision import models
from inference import run_inference, display_results

parser = argparse.ArgumentParser(description='Inference Trained Model')
parser.add_argument('data', metavar='DIR', help='default data path')
parser.add_argument('-bs', '--batch-size', metavar='BS', default=2, help='maximum batchsize')
parser.add_argument('-tp', '--top-predictions', metavar='NUMPRED',\
                     default=5, help='number of top predictions per sample')
parser.add_argument('-exp', '--export', default=False,help='export model to onnx')

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
