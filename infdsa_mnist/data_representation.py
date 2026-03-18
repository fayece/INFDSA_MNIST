import numpy as np
from infdsa_mnist.decision_tree import DecisionTreeClassifier


def quantize(img: np.ndarray) -> np.ndarray:
    """
    Reduce pixel precision from 8-bit to 4-bit
    Memory reduction: 50%
    :param img: Raw uint8 image
    :return: uint8 image with values in range 0–15.
    """
    return (img // 16).astype(np.uint8)


def otsu_binary_threshold(img: np.ndarray) -> np.ndarray:
    """
    Convert image to binary (black and white) using Otsu's method to find optimal threshold
    Memory reduction: 87.5%
    :param img: uint8 image
    :return: Binary image with values 0 or 1.
    """
    best_threshold = 0
    best_variance = float("inf")

    for t in range(1, 256):
        above = img >= t
        below = ~above

        if not np.any(above) or not np.any(below):
            continue

        within_class_variance = (
                np.mean(above) * np.var(img[above]) +
                np.mean(below) * np.var(img[below])
        )

        if within_class_variance < best_variance:
            best_variance = within_class_variance
            best_threshold = t

    return (img >= best_threshold).astype(np.uint8)


def binary_threshold(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Convert pixel values to binary (black and white) based on a fixed threshold.
    The pixels above the threshold become 1, the pixels below become 0.
    Memory reduction: 87.5%
    :param img: uint8 image
    :param threshold: Threshold value for binary conversion (default: 128, midway between 0 and 255).
    :return: Binary image with values 0 or 1.
    """
    return (img > threshold).astype(np.uint8)


def bin_pixels(img: np.ndarray) -> np.ndarray:
    """
    Group pixel values into 4 bins (0-3), representing 2-bit encoding.
    Memory reduction: 75%
    :param img: uint8 image
    :return: uint8 image with values in range 0–3.
    """
    return (img // 64).astype(np.uint8)


def downscale(img: np.ndarray) -> np.ndarray:
    """
    Downscale image by a factor of 2 by averaging 2x2 pixel blocks
    Memory reduction: 75%
    :param img: uint8 image
    :return: uint8 14x14 image
    """
    return img.reshape(14, 2, 14, 2).mean(axis=(1, 3)).astype(np.uint8)


def _upscale(img: np.ndarray) -> np.ndarray:
    """
    Upscale a 14x14 image back to 28x28 using nearest-neighbor interpolation.
    Used internally to restore downscaled images for prediction
    :param img: uint8 14x14 image
    :return: uint8 28x28 image
    """
    return img.repeat(2, axis=0).repeat(2, axis=1)


def sparse_encode(img: np.ndarray) -> np.ndarray:
    """
    Store only non-zero pixels as (row, col, value) triplets.
    Memory reduction varies by image
    :param img: uint8 image
    :return: 2D uint8 array of shape (n_nonzero, 3).
    """
    rows, cols = np.where(img > 0)
    values = img[rows, cols]
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    return np.column_stack((rows, cols, values)).astype(np.uint8)


def _sparse_decode(encoded: np.ndarray) -> np.ndarray:
    """
    Reconstruct image from sparse encoding.
    Used internally to restore sparse images for prediction.
    :param encoded: 2D uint8 array of shape (n_nonzero, 3).
    :return: uint8 image
    """
    img = np.zeros((28, 28), dtype=np.uint8)
    if len(encoded) > 0:
        img[encoded[:, 0], encoded[:, 1]] = encoded[:, 2]
    return img


def sparse_decode(encoded: np.ndarray) -> np.ndarray:
    """
    Public wrapper for sparse decoding.
    :param encoded:
    :return:
    """
    return _sparse_decode(encoded)


def _theoretical_bytes(bits_per_pixel: int, pixels: int = 784) -> float:
    """Returns the theoretical byte size given bits per pixel."""
    return (pixels * bits_per_pixel) / 8


def encoding_memory_bytes(encoded_img: np.ndarray, technique: str) -> float:
    """
    Return the actual or theoretical memory usage in bytes for an encoded image.
    :param encoded_img: Encoded image.
    :param technique: Encoding technique used.
    :return: Memory usage in bytes.
    """
    sizes = {
        "baseline": _theoretical_bytes(8),
        "quantize": _theoretical_bytes(4),
        "binary": _theoretical_bytes(1),
        "binning": _theoretical_bytes(2),
        "downscale": _theoretical_bytes(8, pixels=196),
    }
    if technique == "sparse":
        return float(encoded_img.nbytes)
    return sizes[technique]


def benchmark_encodings(
        clf: DecisionTreeClassifier,
        test_images: np.ndarray,
        test_labels: np.ndarray
) -> dict:
    """
    Benchmark all encoding techniques against the classifier on the full test set.
    Each technique is evaluated on accuracy and average memory usage per image.

    :param clf: A trained DecisionTreeClassifier.
    :param test_images: Test images.
    :param test_labels: Test labels.
    :return: Dict mapping technique name to {'accuracy': float, 'avg_bytes': float}.
    """
    techniques = {
        "Baseline": (
            lambda image: image,
            lambda image: image,
            "baseline"
        ),
        "Quantization (4-bit)": (
            lambda image: quantize(image),
            lambda image: (quantize(image) * 16).astype(np.uint8),
            "quantize"
        ),
        "Binary Threshold (1-bit)": (
            lambda image: binary_threshold(image),
            lambda image: (binary_threshold(image) * 255).astype(np.uint8),
            "binary"
        ),
        "Binning (2-bit)": (
            lambda image: bin_pixels(image),
            lambda image: (bin_pixels(image) * 85).astype(np.uint8),
            "binning"
        ),
        "Downscaling (14x14)": (
            lambda image: downscale(image),
            lambda image: _upscale(downscale(image)),
            "downscale"
        ),
        "Sparse Encoding": (
            lambda image: sparse_encode(image),
            lambda image: _sparse_decode(sparse_encode(image)),
            "sparse"
        ),
    }

    results = {}

    for name, (encode_fn, decode_fn, technique_key) in techniques.items():
        correct = 0
        total_bytes = 0.0

        for img, label in zip(test_images, test_labels):
            encoded = encode_fn(img)
            decoded = decode_fn(img)

            if clf.predict(decoded) == label:
                correct += 1

            total_bytes += encoding_memory_bytes(encoded, technique_key)

        accuracy = correct / len(test_labels)
        avg_bytes = total_bytes / len(test_images)
        results[name] = {"accuracy": accuracy, "avg_bytes": avg_bytes}

    return results



def quantize_downscale(img: np.ndarray) -> np.ndarray:
    """
    Downscales to 14x14 then quantizes to 4-bit.
    Theoretical memory: 98 bytes (4-bit x 196 pixels).
    """
    return quantize(downscale(img))


def binning_downscale(img: np.ndarray) -> np.ndarray:
    """
    Downscales to 14x14 then bins to 2-bit.
    Theoretical memory: 49 bytes (2-bit x 196 pixels).
    """
    return bin_pixels(downscale(img))


def binary_downscale(img: np.ndarray) -> np.ndarray:
    """
    Downscales to 14x14 then binarizes.
    Theoretical memory: ~25 bytes (1-bit x 196 pixels).
    """
    return binary_threshold(downscale(img))


def quantize_sparse(img: np.ndarray) -> np.ndarray:
    """
    Quantizes to 4-bit then sparse encodes non-zero pixels as (row, col, value).
    Memory varies — coordinate overhead still applies, but value range is reduced.
    """
    return sparse_encode(quantize(img))


def binary_sparse(img: np.ndarray) -> np.ndarray:
    """
    Binarizes then stores only the coordinates of 1-pixels as (row, col) pairs.
    No value column needed since all stored pixels are 1.
    Memory: 2 bytes x n_nonzero pixels.
    """
    binary = binary_threshold(img)
    rows, cols = np.where(binary > 0)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.uint8)
    return np.column_stack((rows, cols)).astype(np.uint8)


def binning_sparse(img: np.ndarray) -> np.ndarray:
    """
    Bin to 2-bit, then sparse encode non-zero pixels as (row, col, value).
    Memory varies — coordinate overhead still applies, but value range is reduced.
    """
    return sparse_encode(bin_pixels(img))


def _quantize_downscale_decode(img: np.ndarray) -> np.ndarray:
    """
    Restore quantize+downscale to 28x28 uint8 for prediction.
    """
    return _upscale((img * 16).astype(np.uint8))


def _binning_downscale_decode(img: np.ndarray) -> np.ndarray:
    """
    Restore binning+downscale to 28x28 uint8 for prediction.
    """
    return _upscale((img * 85).astype(np.uint8))


def _binary_downscale_decode(img: np.ndarray) -> np.ndarray:
    """
    Restore binary+downscale to 28x28 uint8 for prediction.
    """
    return _upscale((img * 255).astype(np.uint8))


def _quantize_sparse_decode(encoded: np.ndarray) -> np.ndarray:
    """
    Reconstruct quantize+sparse to 28x28 uint8 for prediction.
    """
    img = np.zeros((28, 28), dtype=np.uint8)
    if len(encoded) > 0:
        img[encoded[:, 0], encoded[:, 1]] = (encoded[:, 2] * 16).astype(np.uint8)
    return img


def _binary_sparse_decode(encoded: np.ndarray) -> np.ndarray:
    """
    Reconstruct binary+sparse (coordinate-only) to 28x28 uint8 for prediction.
    """
    img = np.zeros((28, 28), dtype=np.uint8)
    if len(encoded) > 0:
        img[encoded[:, 0], encoded[:, 1]] = 255
    return img


def _binning_sparse_decode(encoded: np.ndarray) -> np.ndarray:
    """
    Reconstruct binning+sparse to 28x28 uint8 for prediction.
    """
    img = np.zeros((28, 28), dtype=np.uint8)
    if len(encoded) > 0:
        img[encoded[:, 0], encoded[:, 1]] = (encoded[:, 2] * 85).astype(np.uint8)
    return img


def _combination_memory_bytes(encoded: np.ndarray, technique_key: str) -> float:
    theoretical = {
        "quantize_downscale": _theoretical_bytes(4, pixels=196),
        "binning_downscale":  _theoretical_bytes(2, pixels=196),
        "binary_downscale":   _theoretical_bytes(1, pixels=196),
    }
    if technique_key in theoretical:
        return theoretical[technique_key]
    return float(encoded.nbytes)  # sparse-based: actual size varies


def benchmark_combinations(
        clf: DecisionTreeClassifier,
        test_images: np.ndarray,
        test_labels: np.ndarray
) -> dict:
    """
    Benchmarks combination encoding techniques against the classifier.
    Runs separately from benchmark_encodings for a cleaner comparison.

    :param clf: A trained DecisionTreeClassifier.
    :param test_images: Raw uint8 test images.
    :param test_labels: True labels for the test images.
    :return: Dict mapping technique name to {'accuracy': float, 'avg_bytes': float}.
    """
    techniques = {
        "Quantize + Downscale": (
            lambda img: quantize_downscale(img),
            lambda img: _quantize_downscale_decode(quantize_downscale(img)),
            "quantize_downscale"
        ),
        "Binning + Downscale": (
            lambda img: binning_downscale(img),
            lambda img: _binning_downscale_decode(binning_downscale(img)),
            "binning_downscale"
        ),
        "Binary + Downscale": (
            lambda img: binary_downscale(img),
            lambda img: _binary_downscale_decode(binary_downscale(img)),
            "binary_downscale"
        ),
        "Quantize + Sparse": (
            lambda img: quantize_sparse(img),
            lambda img: _quantize_sparse_decode(quantize_sparse(img)),
            "quantize_sparse"
        ),
        "Binary + Sparse": (
            lambda img: binary_sparse(img),
            lambda img: _binary_sparse_decode(binary_sparse(img)),
            "binary_sparse"
        ),
        "Binning + Sparse": (
            lambda img: binning_sparse(img),
            lambda img: _binning_sparse_decode(binning_sparse(img)),
            "binning_sparse"
        ),
    }

    results = {}

    for name, (encode_fn, decode_fn, technique_key) in techniques.items():
        correct = 0
        total_bytes = 0.0

        for img, label in zip(test_images, test_labels):
            encoded = encode_fn(img)
            decoded = decode_fn(img)

            if clf.predict(decoded) == label:
                correct += 1

            total_bytes += _combination_memory_bytes(encoded, technique_key)

        accuracy = correct / len(test_labels)
        avg_bytes = total_bytes / len(test_images)
        results[name] = {"accuracy": accuracy, "avg_bytes": avg_bytes}

    return results
