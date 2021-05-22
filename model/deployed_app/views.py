import os
import onnxruntime as ort
from deployed_app import api
from flask import jsonify, request
from flask_restful import Resource
from onnx_data import preprocess_data, run_inference
from utils import read_imagenet_classnames, display_results


model = ort.InferenceSession('checkpoints/model.onnx')
top_predictions = 2


class ServeInference(Resource):
    def get(self):
        imagenet_classes = read_imagenet_classnames(\
            os.path.join("data", "imagenet_classnames.txt"))
        data = preprocess_data("data")
        predictions = run_inference(model, data, top_predictions)
        predictions = display_results(data, predictions, imagenet_classes, False)
        return predictions

        
api.add_resource(ServeInference, '/app')
