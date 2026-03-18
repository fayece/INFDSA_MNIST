import numpy as np
from infdsa_mnist import mnist_load
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


def find_ambiguous_digits(x_data, y_data, top_n=10, seed=42):
    x_flat = mnist_load.normalize_images(x_data)
    x_flat = mnist_load.flatten_images(x_flat)


    print("Calculating ambiguous digits (this may take a minute)...")

    # Small neural network that uses 50 neurons
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=seed)
    model.fit(x_flat, y_data)

    # Get the probabilities for each class 0 to 1
    probabilities = model.predict_proba(x_flat)

    ambiguous_indices = []

    # Get the confidence gap for each image
    for index, scores in enumerate(probabilities):
        sorted_scores = np.sort(scores)[::-1]
        gap = sorted_scores[0] - sorted_scores[1]

        ambiguous_indices.append((gap, index, scores))

    # Sort by: Most to Least ambiguous
    ambiguous_indices.sort(key=lambda x: x[0])

    result_x = []
    result_y = []

    # Safety net to ensure we don't try to access more indices than available
    actual_top_n = min(top_n, len(ambiguous_indices))

    for i in range(actual_top_n):
        gap, original_index, scores = ambiguous_indices[i]

        top_guess = int(np.argsort(scores)[-1])
        second_guess = int(np.argsort(scores)[-2])

        top_prob = float(scores[top_guess] * 100)
        second_prob = float(scores[second_guess] * 100)
        true_label = int(y_data[original_index])

        label = f"True: {true_label}\n{top_guess} ({top_prob:.0f}%) vs {second_guess} ({second_prob:.0f}%)"

        result_x.append(x_data[original_index])
        result_y.append(label)

    return result_x, result_y


def get_error_matrix(x_data, y_data):
    x_flat = mnist_load.normalize_images(x_data)
    x_flat = mnist_load.flatten_images(x_flat)

    # "Random forest" with 50 decision trees that vote on the answer
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

    # Use cross-validation to get more robust predictions
    predictions = cross_val_predict(model, x_flat, y_data, cv=3, n_jobs=-1)

    return confusion_matrix(y_data, predictions)
