from tensorflow.keras.datasets import mnist


def _load_mnist(npz_path='mnist.npz'):
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=npz_path)
    return (x_train, y_train), (x_test, y_test)


def load_mnist_normalized(npz_path='mnist.npz'):
    (x_train, y_train), (x_test, y_test) = _load_mnist(npz_path)
    return (normalize_images(x_train), y_train), (normalize_images(x_test), y_test)


def load_mnist_flattened(npz_path='mnist.npz'):
    (x_train, y_train), (x_test, y_test) = _load_mnist(npz_path)
    return (flatten_images(x_train), y_train), (flatten_images(x_test), y_test)


def load_mnist_flattened_normalized(npz_path='mnist.npz'):
    (x_train, y_train), (x_test, y_test) = load_mnist_flattened(npz_path)
    return (normalize_images(x_train), y_train), (normalize_images(x_test), y_test)


def normalize_images(x_data):
    return x_data / 255.0


def flatten_images(x_data):
    return x_data.reshape(len(x_data), -1)
