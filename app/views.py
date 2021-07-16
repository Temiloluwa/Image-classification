import requests
import torch
from app import api
from PIL import Image, ImageFile
from io import BytesIO
from flask import jsonify
from flask_restful import Resource, reqparse
from data import preprocess_image
from utils import read_imagenet_classnames, one_prediction, run_inference, parse_base64

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = reqparse.RequestParser(bundle_errors=True)
parser.add_argument('img_url', help='img_url for get query')
parser.add_argument('pred_num', type=int, default=10, help='number of predictions')
model = torch.jit.load("checkpoints/model.pt")
imagenet_classes = read_imagenet_classnames("cache/imagenet_classnames.txt")


class ServeInference(Resource):
    def get(self):
        args = parser.parse_args()
        img_url = args.img_url
        if args.pred_num:
            top_predictions = args.pred_num
        bytes_str = parse_base64(img_url)
        
        if not bytes_str:
            res = requests.get(img_url)
            if res.status_code != 200:
                try:
                    res.raise_for_status()
                except Exception as e:
                    return jsonify({"status": res.status_code,\
                                "msg": str(e)})
            else:
                bytes_str = res.content
            
        im = Image.open(BytesIO(bytes_str))
        im.save("sample.jpg")
        im = preprocess_image(im)
        prediction = run_inference(model, im, top_predictions)
        prediction = one_prediction(prediction, imagenet_classes)
        response = {"status": 200, "msg": prediction}
        return jsonify(response)

api.add_resource(ServeInference, '/app')