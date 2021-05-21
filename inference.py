import torch
import numpy as np
from utils import softmax

def run_inference(model, data, top_predictions, onnx = False):
    input_data, _ = data
    if not onnx:
        predictions = model(input_data)
        probabilities = torch.softmax(predictions, 1)
        pred_indices = torch.argsort(probabilities, 1,\
                            descending=True)[:, : top_predictions]
        probabilities = torch.gather(probabilities, 1, pred_indices)
        probabilities = (probabilities * 100).detach().numpy()
        pred_indices = pred_indices.detach().numpy()
    else:
        input_data = input_data.detach().numpy()
        input_name = model.get_inputs()[0].name
        predictions = model.run(None, {input_name: input_data})[0]
        probabilities = softmax(predictions)
        pred_indices = np.flip(np.argsort\
                        (probabilities, axis=1)[:, -top_predictions:], 1)
        probabilities = np.take_along_axis(probabilities, pred_indices, 1) * 100
    return probabilities, pred_indices


def display_results(data, predictions, imagenet_classes, print_values=True):
    input_data, file_names = data
    probabilities, pred_indices = predictions

    assert len(input_data) == len(file_names)  \
            == len(probabilities) == len(pred_indices),\
            "check batch size of data and prediction"
    
    pred_outputs = {}
    for i, file in enumerate(file_names):
        prob, idx = probabilities[i], pred_indices[i]
        prediction = [f"{imagenet_classes[idx[j]][0]} with probability {prob[j]:0.2f}%" \
                        for j in range(len(prob))]
        prediction = "\n".join(prediction)
        
        if print_values:
            print(f"\n file {file} prediction: {prediction}")
        else:
            pred_outputs[file] = prediction
            if i == len(file_names) - 1:
                return pred_outputs
    