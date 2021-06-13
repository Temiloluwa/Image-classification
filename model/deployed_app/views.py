import os
import requests
import onnxruntime as ort
from deployed_app import api

from PIL import Image
from io import BytesIO
from flask import jsonify, request 
from flask_restful import Resource, reqparse
from onnx_data import preprocess_image, run_inference
from utils import read_imagenet_classnames, one_prediction


parser = reqparse.RequestParser(bundle_errors=True)
parser.add_argument('img_url', type='str', help='img_url for get query')
model = ort.InferenceSession('checkpoints/model.onnx')
imagenet_classes = read_imagenet_classnames(\
            os.path.join("data", "imagenet_classnames.txt"))
top_predictions = 5


class ServeInference(Resource):
    def get(self):
        args = parser.parse_args()
        res = requests.get(args.img_url)
        if res.status_code != 200:
            try:
                res.raise_for_status()
            except Exception as e:
                return jsonify({"status": res.status_code,\
                            "msg": str(e)})

        im = Image.open(BytesIO(res.content))
        im = preprocess_image(im)
        prediction = run_inference(model, im, top_predictions)
        prediction = one_prediction(prediction, imagenet_classes)
        response = {"status": 200, "msg": prediction}
        return jsonify(response)

        
api.add_resource(ServeInference, '/app')
