import numpy as np
import time
from collections import deque

def to_list(x_train, y_train):
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    return [
        [x_train[i], int(y_train[i])]
        for i in range(len(x_train))
    ]


def to_queue(x_train, y_train):
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    return deque([
        [x_train[i], int(y_train[i])]
        for i in range(len(x_train))
    ])


def to_dict(x_train, y_train):
    label_dict = {}

    for image, label in zip(x_train, y_train):
        label = int(label)
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(image)

    return label_dict


def find_first_five_lst(x_train_lst):
    for i, image in enumerate(x_train_lst):
        if image[1] == 5:
            return i
    return None


def add_new_element_lst(x_train_lst, new_image, new_label):
    x_train_lst.append([new_image.tolist(), new_label])


def walk_through_dataset_lst(x_train_lst):
    time_start = time.time()
    for _ in x_train_lst:
        pass

    time_taken = time.time() - time_start
    return f"Walked through the dataset in {time_taken:.2f} seconds."


def remove_first_five_lst(x_train_lst):
    for i, image in enumerate(x_train_lst):
        if image[1] == 5:
            return x_train_lst.pop(i)
    return None


def find_first_five_queue(x_train_queue):
    for i, image in enumerate(x_train_queue):
        if image[1] == 5:
            return i
    return None


def add_new_element_queue(x_train_queue, new_image, new_label):
    x_train_queue.append([new_image.tolist(), int(new_label)])


def walk_through_dataset_queue(x_train_queue):
    time_start = time.time()

    for _ in x_train_queue:
        pass

    time_taken = time.time() - time_start
    return f"Walked through the dataset in {time_taken:.2f} seconds."


def remove_first_five_queue(x_train_queue):
    for _ in range(len(x_train_queue)):
        image = x_train_queue.popleft()

        if image[1] == 5:
            return image

        x_train_queue.append(image)

    return None


def find_first_five_dict(x_train_dict):
    for label, images in x_train_dict.items():
        if label == 5:
            return images[0]
    return None


def add_new_element_dict(x_train_dict, new_image, new_label):
    new_label = int(new_label)

    if new_label not in x_train_dict:
        x_train_dict[new_label] = []

    x_train_dict[new_label].append(new_image)


def walk_through_dataset_dict(x_train_dict):
    time_start = time.time()

    for label, images in x_train_dict.items():
        for _ in images:
            pass

    time_taken = time.time() - time_start
    return f"Walked through the dataset in {time_taken:.2f} seconds."


def remove_first_five_dict(x_train_dict):
    if 5 in x_train_dict and x_train_dict[5]:
        return x_train_dict[5].pop(0)

    return None


def over_ten_thousand_lst(x_train_lst):
    return [image for image in x_train_lst if np.sum(image[0]) > 10000]


def over_ten_thousand_queue(x_train_queue):
    return [image for image in x_train_queue if np.sum(image[0]) > 10000]

def over_ten_thousand_dict(x_train_dict):
    return [image for images in x_train_dict.values() for image in images if np.sum(image) > 10000]