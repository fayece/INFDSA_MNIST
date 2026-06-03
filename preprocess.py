import numpy as np
from PIL import Image


class ImagePreprocessor:

    @staticmethod
    def load(path):
        return np.array(Image.open(path), dtype=np.uint8)

    @staticmethod
    def to_greyscale(img_arr):
        if img_arr.ndim == 2:
            return img_arr
        if img_arr.shape[2] > 3:
            img_arr = img_arr[:, :, :3]  # Keep only RGB channels

        weights = np.array([0.2126, 0.7152, 0.0722])
        return (img_arr @ weights).astype(np.uint8)

    @staticmethod
    def invert(img_arr):
        return 255 - img_arr

    @staticmethod
    def maximize_contrast(img_arr):
        img_min, img_max = img_arr.min(), img_arr.max()
        if img_max > img_min:
            return ((img_arr - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return img_arr

    @classmethod
    def center(cls, img_arr):
        cropped = cls._crop_to_bounding_box(img_arr)
        if cropped is img_arr:
            return img_arr
        return cls._paste_cropped(cropped)

    @staticmethod
    def normalize(img_arr):
        return img_arr / 255.0

    @staticmethod
    def flatten(img_arr):
        return img_arr.reshape(1, -1)

    @staticmethod
    def _crop_to_bounding_box(img_arr):
        threshold = max(10, img_arr.mean())
        coordinates = np.argwhere(img_arr > threshold)

        if len(coordinates) == 0:
            return img_arr
        y0, x0 = coordinates.min(axis=0)
        y1, x1 = coordinates.max(axis=0)
        return img_arr[y0:y1 + 1, x0:x1 + 1]

    @staticmethod
    def _paste_cropped(img_arr):
        canvas = np.zeros((28, 28), dtype=np.uint8)
        h, w = img_arr.shape
        y_offset = (28 - h) // 2
        x_offset = (28 - w) // 2
        canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img_arr
        return canvas

    @staticmethod
    def _should_invert(img_arr, threshold= 127):
        top_left = img_arr[0, 0]
        top_right = img_arr[0, -1]
        bottom_left = img_arr[-1, 0]
        bottom_right = img_arr[-1, -1]

        # Converting all pixel values to Python integers to avoid uint8 overflow issues
        corner_average = (int(top_left) + int(top_right) + int(bottom_left) + int(bottom_right)) / 4
        return corner_average > threshold

    @classmethod
    def process(cls, path):
        img = cls.load(path)
        img = cls.to_greyscale(img)

        if cls._should_invert(img):
            img = cls.invert(img)

        img = cls.maximize_contrast(img)
        img = cls.center(img)
        img = cls.normalize(img)
        return cls.flatten(img)
