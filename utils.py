import base64
import json
import re
import torch.nn.functional as F

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


def display_results(data, predictions, imagenet_classes, print_values=True):
    input_data, file_names = data
    probabilities, pred_indices = predictions

    assert len(input_data) == len(file_names)  \
            == len(probabilities) == len(pred_indices),\
            "check batch size of data and prediction"
    
    pred_outputs = {}
    for i, file in enumerate(file_names):
        prob, idx = probabilities[i], pred_indices[i]
        prediction = [f"{imagenet_classes[idx[j]][0]} with probability {prob[j]:.2f}%" \
                        for j in range(len(prob))]
        prediction = "\n".join(prediction)
        
        if print_values:
            print(f"\n file {file} prediction: {prediction}")
        else:
            pred_outputs[file] = prediction
            if i == len(file_names) - 1:
                return pred_outputs


def one_prediction(prediction, imagenet_classes):
    """ Response for one image """
    prob, idx = prediction[0][0], prediction[1][0]

    return  [{"class": str(imagenet_classes[idx[j]][0]).strip("'"), \
            "probability": str(prob[j]).strip("'")}\
            for j in range(len(prob))]


def run_inference(model, input_data, top_predictions):
    predictions = model(input_data)
    probabilities, pred_indices = F.softmax(predictions, 1).topk(top_predictions)
    probabilities = (probabilities * 100).detach().numpy()
    pred_indices = pred_indices.detach().numpy()
    
    return probabilities, pred_indices


def parse_base64(string_):
    base64_path = "data:image/jpeg;base64,"
    if string_.startswith(base64_path):
        string_ = re.sub(base64_path, "", string_)
        string_ =  bytes(string_, "UTF-8")
        return base64.b64decode(string_)
    else:
        return None
