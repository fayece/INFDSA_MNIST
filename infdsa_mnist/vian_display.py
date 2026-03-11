import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
from .mnist_display import display_grid


def display_single_digit(x_train, y_train, digit, amount=5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    indices = np.where(y_train == digit)[0]
    amount = min(amount, len(indices))
    chosen = np.random.choice(indices, amount, replace=False)

    rows = 2
    cols = math.ceil(amount / rows)

    display_grid(
        x_train[chosen],
        y_train[chosen],
        rows=rows,
        cols=cols,
        title=f"Digit {digit}"
    )


def display_similar_digits(x_train, y_train, samples_per_digit=5, similar_digits=None, subset_size=200):
    if similar_digits is None:
        similar_digits = [(1, 7), (5, 6), (7, 2), (9, 4), (0, 6)]

    num_pairs = len(similar_digits)

    fig = plt.figure(figsize=(samples_per_digit * 2, num_pairs * 4))

    subfigures = fig.subfigures(num_pairs, 1, hspace=0.1)

    if num_pairs == 1:
        subfigures = [subfigures]

    for idx, ((a, b), subfigures) in enumerate(zip(similar_digits, subfigures)):

        subfigures.suptitle(f"Comparison: {a} vs {b}", fontweight='bold')

        axes = subfigures.subplots(2, samples_per_digit)

        imgs_a = x_train[y_train == a][:subset_size].reshape(-1, 784).astype(float)
        imgs_b = x_train[y_train == b][:subset_size].reshape(-1, 784).astype(float)

        dists = cdist(imgs_a, imgs_b, metric='euclidean')

        for i in range(samples_per_digit):
            r, c = np.unravel_index(np.argmin(dists), dists.shape)

            ax_a = axes[0, i]
            ax_a.imshow(imgs_a[r].reshape(28, 28), cmap='gray')
            ax_a.axis('off')
            if i == 0:
                ax_a.set_title(f"Digit {a}")

            ax_b = axes[1, i]
            ax_b.imshow(imgs_b[c].reshape(28, 28), cmap='gray')
            ax_b.axis('off')
            if i == 0:
                ax_b.set_title(f"Digit {b}")

            dists[r, :] = np.inf
            dists[:, c] = np.inf

    plt.show()
