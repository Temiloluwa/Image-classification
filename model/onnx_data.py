import os
import numpy as np
from PIL import Image
from typing import Union
from utils import softmax

def transformations(im):
    im = im.resize((256, 256))
    im = np.array(im) / 255
    mean = np.array([0.485, 0.456, 0.406]).\
        reshape(1, 1, -1)
    std = np.array([0.229, 0.224, 0.225]).\
        reshape(1, 1, -1)
    im = (im - mean)/ std
    im = im.transpose(2, 0, 1)
    im = np.expand_dims(im, 0)
    return im


def preprocess_image(im):
    """ Preprocess image """
    if type(im) is str:
        im =  Image.open(im)
    im = transformations(im).astype("float32")
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

    data_tensors = np.vstack(data_tensors)
    return data_tensors, file_names


def run_inference(model, im, top_predictions):
    input_name = model.get_inputs()[0].name
    predictions = model.run(None, {input_name: im})[0]
    probabilities = softmax(predictions)
    pred_indices = np.flip(np.argsort\
                    (probabilities, axis=1)[:, -top_predictions:], 1)
    probabilities = np.take_along_axis(probabilities, pred_indices, 1) * 100
    return probabilities, pred_indices

