import os
import json
import torch
import numpy as np
import onnx
import torch.onnx as torchonnx

def softmax(x):
    _max = np.max(x, axis=1).reshape(-1, 1)
    x = x - _max
    x = np.exp(x)
    return x/np.sum(x, axis=1, keepdims=True)

def load_json(path: str):
    """Loads json file"""
    with open(path, "r") as f:
        return json.loads(f.read())
    
def read_imagenet_classnames(path:str):
    with open(path, "r") as f:
        temp = f.readlines()

    temp = [i.strip(" ").strip("\n").strip(", ").split(":") \
                        for i in temp]
    temp = {int(k):v.strip(" ").split(",") for k,v in temp}
    classes = []
    for i in temp:
        classes.append([k.strip(" ") for k in temp[i]])
    return classes

def export_to_onnx(model):
    model_path = 'checkpoints/model.onnx'
    sample_input = torch.randn((1, 3, 256, 256))
    torchonnx.export(model, sample_input, model_path)
    model = onnx.load(model_path)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
    onnx.save(model, model_path)
