from infdsa_mnist import helpers
from .mnist_display import display_grid
import numpy as np
import math



def show_image(x_train, y_train, amount=5, seed=42, rotate=False):
    if seed is not None:
        np.random.seed(seed)

    amounts = np.random.choice(len(x_train), size=amount, replace=False)
    images = x_train[amounts]
    labels = y_train[amounts]

    if rotate:
        images = np.rot90(images, k=1, axes=(1, 2))

    display_grid(
        images,
        labels,
        rows=2,
        cols=math.ceil(amount / 2),
        title="Random Training Samples")


def image_count(x_train, x_test):
    train = len(x_train)
    test = len(x_test)
    total = train + test
    return train, test, total


def digit_distribution(y_train, y_test):
    distribution = {i: 0 for i in range(10)}
    for label in y_train:
        distribution[label] += 1
    for label in y_test:
        distribution[label] += 1

    return distribution


def is_balanced(distribution, threshold=1.5):
    try:
        threshold = float(threshold)

        if threshold <= 0:
            raise ValueError("Threshold must be positive")
    except ValueError:
        print("Invalid input. Using default threshold of 1.5 instead.")
        threshold = 1.5

    counts = distribution.values()
    balanced = max(counts) / min(counts) < threshold

    return "The dataset is balanced." if balanced else "The dataset is not balanced."


def image_shape_and_dtype(x_train):
    shape = x_train[0].shape
    dtype = x_train[0].dtype

    shape_str = "x".join(map(str, shape)) + " px"

    dtype_str = helpers.human_readable_dtype(dtype)

    return shape_str, dtype_str


def memory_usage(x_train, x_test):
    single_image_memory = x_train[0].nbytes
    total_memory = x_train.nbytes + x_test.nbytes
    return single_image_memory, total_memory


def explore_dataset(x_train, y_train, x_test, y_test, user_threshold=None):
    shape, dtype = image_shape_and_dtype(x_train)
    train_count, test_count, total_count = image_count(x_train, x_test)
    distribution = digit_distribution(y_train, y_test)
    is_balanced_result = is_balanced(distribution, user_threshold or ask_for_threshold())
    return shape, dtype, train_count, test_count, total_count, distribution, is_balanced_result