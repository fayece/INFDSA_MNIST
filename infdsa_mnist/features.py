import numpy as np


def get_bounding_box(img, threshold=0.0):
    """
    Helper function to find the edges of the digit in the image.
    :return: Tuple of (min_row, max_row, min_col, max_col) or None if empty.
    """
    rows, cols = np.where(img > threshold)
    if len(rows) == 0:
        return None

    return (
        int(np.min(rows)),
        int(np.max(rows)),
        int(np.min(cols)),
        int(np.max(cols)))


def average_pixel_intensity(img) -> float:
    """
    Calculates the average pixel intensity for a single image.
    :param img: 2D NumPy array (e.g., 28x28)
    :return: Scalar mean value
    """
    return float(np.mean(img))


def symmetry(img) -> float:
    """
    Calculates the degree of symmetry for a single image.
    :param img: 2D NumPy array
    :return: Normalized symmetry score (0.0 to 1.0)
    """
    bbox = get_bounding_box(img, threshold=0.0)
    if bbox is None:
        return 0.0

    min_row, max_row, min_col, max_col = bbox
    img_crop = img[min_row:max_row + 1, min_col:max_col + 1]

    vertical_flip = np.flip(img_crop, axis=1)
    horizontal_flip = np.flip(img_crop, axis=0)

    vertical_error = np.mean(np.abs(img_crop - vertical_flip))
    horizontal_error = np.mean(np.abs(img_crop - horizontal_flip))

    average_error = (vertical_error + horizontal_error) / 2

    max_pixel_value = 1.0
    symmetry_score = 1.0 - (average_error / max_pixel_value)

    return float(symmetry_score)


def center_point_concentration(img) -> float:
    """
    Calculates the average intensity in the central 4x4 area.
    :param img: 2D NumPy array (28x28)
    :return: Scalar concentration value
    """
    center_region = img[12:16, 12:16]
    return float(np.mean(center_region))


def aspect_ratio(img) -> float:
    """
    Calculates the width/height ratio of the digit's bounding box.
    :param img: 2D NumPy array
    :return: Float (width / height)
    """
    bbox = get_bounding_box(img, threshold=0.01)
    if bbox is None:
        return 1.0

    min_row, max_row, min_col, max_col = bbox

    height = (max_row - min_row) + 1
    width = (max_col - min_col) + 1

    return float(width / height)


def region_intensities(img) -> dict:
    """
    Calculates the average pixel intensity in the four quadrants of the image.
    :param img: 2D NumPy array (28x28)
    :return: Scalar region intensity value
    """
    top_left = img[0:14, 0:14]
    top_right = img[0:14, 14:28]
    bottom_left = img[14:28, 0:14]
    bottom_right = img[14:28, 14:28]

    avg_top_left = np.mean(top_left)
    avg_top_right = np.mean(top_right)
    avg_bottom_left = np.mean(bottom_left)
    avg_bottom_right = np.mean(bottom_right)

    return {
        "tl_intensity": float(avg_top_left),
        "tr_intensity": float(avg_top_right),
        "bl_intensity": float(avg_bottom_left),
        "br_intensity": float(avg_bottom_right)
    }


def extract_features(img):
    """
    Extracts all features for a single image into a matrix
    Safely handles both unnormalized (uint8) and normalized (float) images.
    :param img: A 2D NumPy array (28x28)
    :return: A matrix of feature values
    """
    if img.dtype == np.uint8:
        img_norm = img / 255.0
    else:
        img_norm = img

    regions = region_intensities(img_norm)

    return np.array([
        average_pixel_intensity(img_norm),
        symmetry(img_norm),
        center_point_concentration(img_norm),
        aspect_ratio(img_norm),
        regions["tl_intensity"],
        regions["tr_intensity"],
        regions["bl_intensity"],
        regions["br_intensity"]
    ], dtype=np.float32)
