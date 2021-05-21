import os
import onnxruntime as ort
from deployed_app import api
from flask import jsonify, request
from flask_restful import Resource
from data import preprocess_data
from inference import run_inference, display_results
from utils import read_imagenet_classnames


model = ort.InferenceSession('checkpoints/model.onnx')
max_predictions = 2


class ServeInference(Resource):
    def get(self):
        imagenet_classes = read_imagenet_classnames(\
            os.path.join("data", "imagenet_classnames.txt"))
        data = preprocess_data("data")
        predictions = run_inference(model, data, max_predictions, onnx=True)
        predictions = display_results(data, predictions, imagenet_classes, False)
        return predictions

        
api.add_resource(ServeInference, '/app')
