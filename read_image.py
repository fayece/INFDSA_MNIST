import numpy as np
from preprocess import ImagePreprocessor
from mysterydevice_model import NeuralNetworkInference
import gc

preprocessor = ImagePreprocessor()

input_nodes = 28 * 28
output_nodes = 10
total_models = 12
hidden_layer_sizes = [[110 + (i * 3)] for i in range(total_models)]

def load_image(filename):
    return preprocessor.process(filename)


def classify_image(image, threshold=0.85):
    all_probs = []
    for i in range(total_models):
        m = NeuralNetworkInference(f'weights_{i}.npz')
        _, probs = m.predict(image)
        all_probs.append(probs.flatten().copy())
        del m
        gc.collect()

    probs = np.mean(all_probs, axis=0)

    best_indices = np.argsort(probs)
    highest_idx = best_indices[-1]
    second_idx = best_indices[-2]

    highest_prob = probs[highest_idx]
    second_prob = probs[second_idx]

    margin = highest_prob - second_prob

    if highest_prob < threshold or margin < 0.8:
        primary_pred = "niet herkend"
    else:
        primary_pred = str(highest_idx)

    guess_if_forced = str(highest_idx)

    return primary_pred, highest_prob, guess_if_forced, second_idx, second_prob
