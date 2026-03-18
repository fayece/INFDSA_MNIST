import sys
import pickle
import numpy as np
from infdsa_mnist.tree_node import TreeNode

class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self._root: TreeNode | None = None

        self.train_accuracy: float | None = None
        self.node_count: int | None = None
        self.depth_used: int | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._root = self._build_tree(features, labels, depth=0)
        self.depth_used = self.max_depth
        self.node_count = self._count_nodes(self._root)

    @classmethod
    def fit_best_depth(
            cls,
            features: np.ndarray,
            labels: np.ndarray,
            test_images: np.ndarray,
            test_labels: np.ndarray,
            depth_range: range = range(1, 16)
    ) -> "DecisionTreeClassifier":
        """
        Trains a decision tree model and finds the optimal depth for the tree.

        :param features: Pre-extracted float32 feature array (n_samples, 8)
        :param labels: Training labels
        :param test_images: Raw uint8 test images for evaluation
        :param test_labels: True labels for the test images
        :param depth_range: Range of depths to try (default: 1–15)
        :return: Best fitted DecisionTreeClassifier instance
        """
        best_clf = None
        best_accuracy = -1.0
        results = {}

        for depth in depth_range:
            clf = cls(max_depth=depth)
            clf.fit(features, labels)
            accuracy = clf.evaluate(test_images, test_labels)
            results[depth] = accuracy
            print(f"Depth {depth:>2}: {accuracy * 100:.2f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_clf = clf

        best_clf.train_accuracy = best_accuracy
        print(f"\nBest depth: {best_clf.depth_used} ({best_accuracy * 100:.2f}% accuracy)")
        return best_clf

    def predict(self, img: np.ndarray) -> int:
        """
        Predicts the label for a single image.
        :param img: A single 2D uint8 NumPy array (28x28)
        :return: Predicted label (0–9)
        """
        if self._root is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        features = extract_features(img)
        return self._traverse(self._root, features)

    def evaluate(self, test_images: np.ndarray, test_labels: np.ndarray) -> float:
        """
        Evaluates the model on a set of test images.

        :param test_images: Array of raw uint8 images
        :param test_labels: Array of true labels
        :return: Tuple of (accuracy as a float 0.0–1.0, list of predicted labels)
        """
        correct = sum(
            self.predict(img) == label
            for img, label in zip(test_images, test_labels)
        )
        return correct / len(test_labels)

    def save(self, path: str) -> None:
        """
        Serializes the trained model to disk using pickle.

        :param path: File path to save the model to (e.g., 'optimal_tree.pkl')
        """
        if self._root is None:
            raise RuntimeError("Cannot save an untrained model. Call fit() first.")

        with open(path, "wb") as f:
            pickle.dump(self, f)

        size_kb = self._pickle_size_kb()
        print(f"Saved to '{path}' ({size_kb:.2f} KB)")

    @classmethod
    def load(cls, path: str) -> "DecisionTreeClassifier":
        """
        Loads a serialized model from the disk.

        :param path: File path of the saved model
        :return: Loaded DecisionTreeClassifier instance
        """
        with open(path, "rb") as f:
            clf = pickle.load(f)

        if not isinstance(clf, cls):
            raise TypeError(f"Expected DecisionTreeClassifier, got {type(clf)}")

        print(f"Loaded classifier (depth={clf.depth_used}, nodes={clf.node_count})")
        return clf

    def ram_usage_kb(self) -> float:
        """Estimates the in-memory size of the tree in KB."""
        return self._get_tree_ram_bytes(self._root) / 1024

    def storage_size_kb(self) -> float:
        """Returns the pickle-serialized size of the full classifier in KB."""
        return self._pickle_size_kb()

    def _build_tree(self, features: np.ndarray, labels: np.ndarray, depth: int) -> TreeNode | None:
        """
        Recursively builds a decision tree.
        :param features: 2D NumPy array of features
        :param labels: 1D NumPy array of labels
        :param depth: Current depth of the tree
        :return: A TreeNode object representing either a leaf node or a decision node
        """
        if len(labels) == 0:
            return None

        unique_labels, counts = np.unique(labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]

        # Base cases: pure node or max depth reached
        if len(unique_labels) == 1 or depth >= self.max_depth:
            return TreeNode(prediction=majority_label)

        best_feature_idx: int = -1
        best_threshold: float = 0.0
        best_score: float = -1

        for i in range(features.shape[1]):
            threshold = np.mean(features[:, i])
            left_mask = features[:, i] < threshold
            right_mask = ~left_mask

            if not np.any(left_mask) or not np.any(right_mask):
                continue

            left_majority = int(np.max(np.unique(labels[left_mask], return_counts=True)[1]))
            right_majority = int(np.max(np.unique(labels[right_mask], return_counts=True)[1]))
            score = left_majority + right_majority

            if score > best_score:
                best_score = score
                best_feature_idx = i
                best_threshold = float(threshold)

        # No valid split found — return a leaf
        if best_score == -1:
            return TreeNode(prediction=majority_label)

        left_mask = features[:, best_feature_idx] < best_threshold
        right_mask = ~left_mask

        return TreeNode(
            feature_idx=best_feature_idx,
            threshold=best_threshold,
            left=self._build_tree(features[left_mask], labels[left_mask], depth + 1),
            right=self._build_tree(features[right_mask], labels[right_mask], depth + 1)
        )

    def _traverse(self, node: TreeNode, feature_values: np.ndarray) -> int:
        current = node
        while current.prediction is None:
            if feature_values[current.feature_idx] < current.threshold:
                current = current.left
            else:
                current = current.right
        return current.prediction

    def _count_nodes(self, node: TreeNode | None) -> int:
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def _get_tree_ram_bytes(self, node: TreeNode | None) -> int:
        if node is None:
            return 0
        size = sys.getsizeof(node) + sys.getsizeof(node.__dict__)
        return size + self._get_tree_ram_bytes(node.left) + self._get_tree_ram_bytes(node.right)

    def _pickle_size_kb(self) -> float:
        return len(pickle.dumps(self)) / 1024


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
    :param img: A 2D NumPy array (28x28)
    :return: A matrix of feature values
    """
    img_norm = img / 255.0

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

def calculate_system_metrics(subset, clf: DecisionTreeClassifier) -> dict:
    dataset_ram_kb = (
        subset["images"].nbytes +
        subset["labels"].nbytes +
        subset["features"].nbytes
    ) / 1024

    return {
        "dataset_ram_kb": dataset_ram_kb,
        "total_nodes": clf.node_count,
        "tree_ram_kb": clf.ram_usage_kb(),
        "total_ram_kb": dataset_ram_kb + clf.ram_usage_kb(),
        "tree_storage_kb": clf.storage_size_kb(),
        "dataset_storage_kb": dataset_ram_kb,
        "total_storage_kb": clf.storage_size_kb() + dataset_ram_kb
    }
