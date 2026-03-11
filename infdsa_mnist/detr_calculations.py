import sys
import pickle
import numpy as np
from infdsa_mnist.tree_node import TreeNode


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
        "tl_intensity": float(np.mean(top_left)),
        "tr_intensity": float(np.mean(top_right)),
        "bl_intensity": float(np.mean(bottom_left)),
        "br_intensity": float(np.mean(bottom_right))
    }


def extract_features(img):
    """
    Extracts all features for a single image into a matrix
    :param img: A 2D NumPy array (28x28)
    :return: A matrix of feature values
    """
    img = img / 255.0

    regions = region_intensities(img)

    return np.array([
        average_pixel_intensity(img),
        symmetry(img),
        center_point_concentration(img),
        aspect_ratio(img),
        regions["tl_intensity"],
        regions["tr_intensity"],
        regions["bl_intensity"],
        regions["br_intensity"]
    ])


def create_subset(x_train, y_train, subset_size=200) -> dict:
    """
    Creates a subset of the training data for model training.

    :param x_train: Array of training images
    :param y_train: Array of training labels
    :param subset_size: The default subset size is 200 to keep a small dataset for embedded models.
    :return: Dictionary containing pre-allocated NumPy arrays for training and validation data
    """
    dataset = {
        "images": np.zeros((subset_size, 28, 28), dtype=np.uint8),
        "labels": np.zeros(subset_size, dtype=np.uint8),
        "features": np.zeros((subset_size, 8), dtype=np.float32)
    }

    for i in range(subset_size):
        img = x_train[i]
        label = y_train[i]

        features = extract_features(img)

        dataset["images"][i] = img
        dataset["labels"][i] = label
        dataset["features"][i] = features

    return dataset


def build_tree(features, labels, depth=0, max_depth=4) -> TreeNode | None:
    """
    Recursively builds a simple decision tree.
    :param features: 2D NumPy array of features
    :param labels: 1D NumPy array of labels
    :param depth: Current depth of the tree
    :param max_depth: Maximum depth, to preserve RAM constraints
    :return: A TreeNode object representing either a leaf node or a decision node
    """

    if len(labels) == 0:
        return None

    unique_labels, counts = np.unique(labels, return_counts=True)
    majority_label = unique_labels[np.argmax(counts)]

    if len(unique_labels) == 1 or depth >= max_depth:
        return TreeNode(prediction=majority_label)

    best_feature_idx = -1
    best_threshold = 0
    best_score = -1

    num_features = features.shape[1]

    for i in range(num_features):
        threshold = np.mean(features[:, i])

        left_mask = features[:, i] < threshold
        right_mask = ~left_mask  # Invert the mask

        if not np.any(left_mask) or not np.any(right_mask):
            continue

        left_labels = labels[left_mask]
        right_labels = labels[right_mask]

        left_majority_count = int(np.max(np.unique(left_labels, return_counts=True)[1]))
        right_majority_count = int(np.max(np.unique(right_labels, return_counts=True)[1]))



        score = left_majority_count + right_majority_count

        if score > best_score:
            best_score = score
            best_feature_idx = i
            best_threshold = threshold

    if best_score == -1:
        return TreeNode(prediction=majority_label)

    left_mask = features[:, best_feature_idx] < best_threshold
    right_mask = ~left_mask

    left_child = build_tree(features[left_mask], labels[left_mask], depth + 1, max_depth)
    right_child = build_tree(features[right_mask], labels[right_mask], depth + 1, max_depth)

    return TreeNode(feature_idx=best_feature_idx, threshold=best_threshold, left=left_child, right=right_child)


def predict(node, feature_values) -> int:
    """
    Traverses the tree to predict the label.

    :param node: The root TreeNode of the decision tree
    :param feature_values: A 1D NumPy array of the extracted features for the image
    :return: The predicted label (0-9)
    """
    current_node = node
    while current_node.prediction is None:
        val = feature_values[current_node.feature_idx]
        if val < current_node.threshold:
            current_node = current_node.left
        else:
            current_node = current_node.right

    return current_node.prediction


def test_tree(tree_node, test_images, test_labels, num_samples=10) -> tuple[int, list[int]]:
    """
    Tests the decision tree on unseen data and prints the results.

    :param tree_node: The root TreeNode of the decision tree
    :param test_images: A NumPy array of unseen images
    :param test_labels: A NumPy array of the true labels for the unseen images
    :param num_samples: How many images to test the tree on (default: 10)
    :return: A tuple containing the number of correct guesses and a list of predicted labels
    """
    correct_guesses = 0
    predictions = []

    for i, img in enumerate(test_images[:num_samples]):
        actual_label = test_labels[i]
        features = extract_features(img)
        predicted_label = predict(tree_node, features)

        predictions.append(predicted_label)

        if predicted_label == actual_label:
            correct_guesses += 1

    return correct_guesses, predictions

def count_tree_nodes(node) -> int:
    """
    Counts the number of nodes in a decision tree.

    :param node: The root TreeNode of the decision tree
    :return: The total number of nodes in the tree
    """
    if node is None:
        return 0
    return 1 + count_tree_nodes(node.left) + count_tree_nodes(node.right)


def get_tree_ram_bytes(node) -> int:
    """
    Calculates the RAM usage of the tree in bytes.

    :param node: The root TreeNode of the decision tree
    :return: The estimated memory usage of the tree in bytes
    """
    if node is None:
        return 0
    current_node_size = sys.getsizeof(node) + sys.getsizeof(node.__dict__)
    return current_node_size + get_tree_ram_bytes(node.left) + get_tree_ram_bytes(node.right)


def calculate_system_metrics(subset, optimal_tree):
    """
    Creates a detailed report of the system's RAM and storage usage, as to validate the embedded hardware constraints.

    :param subset: The subset of data used for training and testing the decision tree.
    :param optimal_tree: The decision tree model that has been trained and optimized for the given subset.
    :return: None
    """

    dataset_ram_kb = (subset["images"].nbytes + subset["labels"].nbytes + subset["features"].nbytes) / 1024
    total_nodes = count_tree_nodes(optimal_tree)
    tree_ram_kb = get_tree_ram_bytes(optimal_tree) / 1024
    total_ram_kb = dataset_ram_kb + tree_ram_kb

    tree_storage_kb = len(pickle.dumps(optimal_tree)) / 1024
    dataset_storage_kb = dataset_ram_kb
    total_storage_kb = tree_storage_kb + dataset_storage_kb

    return {
        "dataset_ram_kb": dataset_ram_kb,
        "total_nodes": total_nodes,
        "tree_ram_kb": tree_ram_kb,
        "total_ram_kb": total_ram_kb,
        "tree_storage_kb": tree_storage_kb,
        "dataset_storage_kb": dataset_storage_kb,
        "total_storage_kb": total_storage_kb
    }