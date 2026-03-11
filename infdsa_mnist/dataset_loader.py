from tensorflow.keras.datasets import mnist
import numpy as np


def load_mnist(npz_path='mnist.npz'):
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=npz_path)
    return (x_train, y_train), (x_test, y_test)


def combine_data(x_train, y_train, x_test, y_test, include_train=True, include_test=True):
    if not include_train and not include_test:
        raise ValueError("Cannot calculate without any included data.")

    if include_train and include_test:
        x_data = np.concatenate((x_train, x_test), axis=0)
        y_data = np.concatenate((y_train, y_test), axis=0)
    elif include_train:
        x_data = x_train
        y_data = y_train
    else:
        x_data = x_test
        y_data = y_test

    return x_data, y_data


def load_mnist_normalized(npz_path='mnist.npz'):
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=npz_path)
    return (normalize_images(x_train), y_train), (normalize_images(x_test), y_test)


def flatten_images(x_data):
    return x_data.reshape(len(x_data), -1)


def normalize_images(x_data):
    return x_data / 255.0


def flatten_one(img):
    return img.reshape(-1)


def normalize_one(img):
    return img / 255.0
