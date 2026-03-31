import numpy as np
import time
from sklearn.cluster import KMeans
from infdsa_mnist.encoding import bin_pixels


np.random.seed(70000)  # Using 70000 as the seed for reproducibility, based on the fact the MNIST dataset has 70,000 total images.


def make_prototype(x_train, y_train, prototypes_per_digit=10, use_sklearn = False):
    all_digits = list(set(y_train))

    prototypes = []
    labels = []

    for digit in all_digits:
        digit_images = x_train[y_train == digit]

        if use_sklearn:
            flat_images = digit_images.reshape(len(digit_images), -1)
            km = KMeans(n_clusters=prototypes_per_digit, n_init='auto', random_state=0)
            km.fit(flat_images)
            digit_prototypes = km.cluster_centers_.reshape(prototypes_per_digit, 28, 28)
        else:
            flat_digit_images = digit_images.reshape(len(digit_images), -1)
            initial_prototypes = find_optimal_centroid_positioning(flat_digit_images, k=prototypes_per_digit)
            flat_prototypes = k_means(flat_digit_images, initial_prototypes)
            digit_prototypes = flat_prototypes.reshape(prototypes_per_digit, 28, 28)

        prototypes.extend(digit_prototypes)
        labels.extend([digit] * prototypes_per_digit)

    return np.array(prototypes), np.array(labels)


def find_optimal_centroid_positioning(points, k=10):
    points = np.array(points)

    centroids = [points[np.random.randint(len(points))]]

    points_sq = np.sum(points ** 2, axis=1, keepdims=True)

    for _ in range(1, k):
        curr_centroids = np.array(centroids)

        # Instead of calculating the Euclidean distance, we use the squared distance to avoid a square root operation
        # This allows us to use .dot(), which is much faster than looping through each point and centroid for distance calculation.
        centroids_sq = np.sum(curr_centroids ** 2, axis=1)
        dot_product = np.dot(points, curr_centroids.T)

        # Using .maximum(0, ...) ensures floating point inaccuracies don't result in small negative distances
        distances_sq = np.maximum(0, points_sq + centroids_sq - 2 * dot_product)

        min_distances_sq = np.min(distances_sq, axis=1)

        total_dist = np.sum(min_distances_sq)
        if total_dist == 0:
            break

        # Points further away have a higher chance of being chosen as centroids, stimulating a more even distribution of centroids.
        probabilities = min_distances_sq / total_dist

        next_centroid = np.random.choice(len(points), p=probabilities)
        centroids.append(points[next_centroid])

    return np.array(centroids)


def k_means(points, centroids, max_iterations=100):
    points = np.array(points)
    centroids = np.array(centroids)
    flat_points = points.reshape(len(points), -1)

    points_sq = np.sum(flat_points ** 2, axis=1, keepdims=True)

    for _ in range(max_iterations):
        k = len(centroids)
        flat_centroids = centroids.reshape(k, -1)

        # Using an algebra trick to keep calculations within a 2D array instead of calculating in 3D space
        centroids_sq = np.sum(flat_centroids ** 2, axis=1)
        dot_product = np.dot(flat_points, flat_centroids.T)
        distances_sq = points_sq + centroids_sq - 2 * dot_product

        # We don't need to sqrt, since the smallest squared distance is also the smallest sqrt distance
        nearest_label = np.argmin(distances_sq, axis=1)

        new_centroids_l = []
        for i in range(k):
            cluster_points = points[nearest_label == i]
            if len(cluster_points) > 0:
                new_centroids_l.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids_l.append(centroids[i])

        new_centroids = np.array(new_centroids_l)

        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return centroids


def get_distance(to_compare, centroid):
    return np.sum(np.abs(to_compare - centroid))


def predict(unseen_digit, centroids, labels):
    flat_digit = unseen_digit.flatten()
    flat_centroids = centroids.reshape(len(centroids), -1)

    # Calculate the Euclidian distance between the unseen digit and each centroid
    distances = np.linalg.norm(flat_centroids - flat_digit, axis=1)

    closest_index = np.argmin(distances)
    return labels[closest_index]

def predict_all(unseen_digits, centroids, labels):
    flat_digits = unseen_digits.reshape(len(unseen_digits), -1)
    flat_centroids = centroids.reshape(len(centroids), -1)

    # [:, None] creates a dummy dimension, expanding flat_digits from 2D to 3D
    # NumPy broadcasts the subtraction, comparing all test images against all centroids simultaneously.
    distances = np.linalg.norm(flat_digits[:, None] - flat_centroids, axis=2)
    closest_indices = np.argmin(distances, axis=1)

    return labels[closest_indices]


def create_confusion_matrix(y_test, y_pred):
    cm = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_test, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def evaluate_model(train_data, test_data, k=10, use_sklearn=False):
    x_train, y_train = train_data
    x_test, y_test = test_data

    start_time = time.time()
    prototypes, labels = make_prototype(x_train, y_train, prototypes_per_digit=k, use_sklearn=use_sklearn)

    predictions = predict_all(x_test, prototypes, labels)

    exec_time = time.time() - start_time
    accuracy = np.mean(predictions == y_test)
    memory_kb = prototypes.nbytes / 1024
    cm = create_confusion_matrix(y_test, predictions)

    return {
        "k": k,
        "accuracy": accuracy,
        "memory_kb": memory_kb,
        "time_seconds": exec_time,
        "prototypes": prototypes,
        "labels": labels,
        "confusion_matrix": cm,
    }


def evaluate_binned_model(train_data, test_data, k=20, use_sklearn=False):
    import time
    x_train, y_train = train_data
    x_test, y_test = test_data

    start_time = time.time()

    prototypes, labels = make_prototype(x_train, y_train, prototypes_per_digit=k, use_sklearn=use_sklearn)
    raw_memory_kb = prototypes.nbytes / 1024

    uint8_prototypes = (prototypes * 255).astype(np.uint8)
    binned_prototypes = np.array([bin_pixels(p) for p in uint8_prototypes])

    binned_memory_bytes = (binned_prototypes.size * 2) / 8
    binned_memory_kb = binned_memory_bytes / 1024

    uint8_test_images = (x_test * 255).astype(np.uint8)
    binned_x_test = np.array([bin_pixels(img) for img in uint8_test_images])

    active_int32_buffers_bytes = (784 * 4) + (784 * 4)
    peak_inference_ram_kb = binned_memory_kb + (active_int32_buffers_bytes / 1024)

    predictions = predict_streaming(binned_x_test, binned_prototypes, labels)
    accuracy = np.mean(predictions == y_test)

    exec_time = time.time() - start_time

    return {
        "k": k,
        "raw_memory_kb": raw_memory_kb,
        "binned_memory_kb": binned_memory_kb,
        "peak_inference_ram_kb": peak_inference_ram_kb,
        "binned_predictions": predictions,
        "binned_accuracy": accuracy,
        "time_seconds": exec_time
    }


def predict_streaming(unseen_digits_uint8, prototypes_uint8, labels):
    predictions = []

    for digit in unseen_digits_uint8:
        flat_digit = digit.flatten().astype(np.int32)

        min_dist = float('inf')
        best_label = -1

        for i, proto in enumerate(prototypes_uint8):
            flat_proto = proto.flatten().astype(np.int32)

            dist = np.sum((flat_proto - flat_digit) ** 2)

            if dist < min_dist:
                min_dist = dist
                best_label = labels[i]

        predictions.append(best_label)

    return np.array(predictions)