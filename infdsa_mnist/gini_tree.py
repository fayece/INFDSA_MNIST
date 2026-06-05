from infdsa_mnist.features import extract_features
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def train_gini_tree(x_train, y_train):
    dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=20,
        random_state=42
    )

    dt.fit(x_train, y_train)

    return dt


def train_gini_tree_ch3(x_train, y_train):
    x_features = np.array([extract_features(img) for img in x_train])

    dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=20,
        random_state=42
    )

    dt.fit(x_features, y_train)

    return dt


def train_gini_tree_combined(x_train, y_train):
    x_train_flat = x_train.reshape(x_train.shape[0], -1)

    x_custom = np.array([extract_features(img) for img in x_train])
    x_combined = np.hstack((x_train_flat, x_custom))

    dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=20,
        random_state=42
    )

    dt.fit(x_combined, y_train)

    return dt


def accuracy(y_test, y_pred):
    return np.mean(y_test == y_pred)


def create_confusion_matrix(y_true, y_pred):
    cm = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm
