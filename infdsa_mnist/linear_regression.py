from sklearn.linear_model import LinearRegression
from infdsa_mnist.decision_tree import extract_features as _extract_features
import numpy as np


def make_binary_labels(y, digit):
    return (y == digit).astype(int)


def train_linear_model(x_train, y_binary):
    model = LinearRegression()
    model.fit(x_train, y_binary)
    return model


def make_models(x_train, y_train):
    models = {}
    for digit in range(10):
        binary_labels = make_binary_labels(y_train, digit)
        models[digit] = train_linear_model(x_train, binary_labels)
    return models


def predict_num(model, image):
    return model.predict([image])[0]


def predict_case(models, image):
    scores = {digit: predict_num(model, image) for digit, model in models.items()}
    return max(scores, key=scores.get)


def predict_cases(models, images):
    scores = np.array([model.predict(images) for model in models.values()])
    return np.argmax(scores, axis=0)


def combine_features(*features):
    return np.hstack(features)


def extract_features(images):
    return np.array([
        _extract_features(img) for img in images
    ])


def create_confusion_matrix(y_true, y_pred):
    cm = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100