import numpy as np


def average_pixel_value(x_data):
    return np.mean(x_data)


def average_pixel_values_per_digit(x_data, y_data):
    averages = {}
    for digit in range(10):
        digit_images = x_data[y_data == digit]
        averages[digit] = np.mean(digit_images)
    return averages


def calculate_average_digit_images(x_data, y_data):
    average_images = {}
    for digit in range(10):
        digit_images = x_data[y_data == digit]
        average_images[digit] = np.mean(digit_images, axis=0)
    return average_images


def standard_deviation_pixel_values_per_digit(x_data, y_data):
    std_images = {}
    for digit in range(10):
        digit_images = x_data[y_data == digit]
        std_images[digit] = np.std(digit_images, axis=0)

    return std_images


def average_pixel_value_across_dataset(x_data):
    x_data = x_data.astype(np.float32)

    if x_data.ndim == 4:
        x_data = np.squeeze(x_data, axis=-1)
    avg_pixels = np.mean(x_data, axis=0)

    return avg_pixels


def standard_deviation_pixel_value_across_dataset(x_data):
    x_data = x_data.astype(np.float32)

    if x_data.ndim == 4:
        x_data = np.squeeze(x_data, axis=-1)
    std_pixels = np.std(x_data, axis=0)

    return std_pixels
