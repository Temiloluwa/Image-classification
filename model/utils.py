import json
import numpy as np


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