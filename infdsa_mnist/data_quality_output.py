import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import infdsa_mnist.mnist_output as mnist_display


def display_error_heatmap(cm, title="Computer Classification Errors"):
    plt.figure(figsize=(10, 8))

    diagonal_mask = np.eye(10, dtype=bool)

    ax = plt.gca()
    ax.set_facecolor('black')
    ax.grid(False)

    sns.heatmap(cm, mask=diagonal_mask, annot=True, fmt="d", cmap="Oranges", cbar=True, linewidths=0, ax=ax)

    plt.title(title, fontsize=18, pad=20)
    plt.xlabel("What the Computer Guessed", fontsize=14, labelpad=10)
    plt.ylabel("What the Digit Was Labeled As", fontsize=14, labelpad=10)

    plt.xticks(ticks=[i + 0.5 for i in range(10)], labels=[str(i) for i in range(10)])
    plt.yticks(ticks=[i + 0.5 for i in range(10)], labels=[str(i) for i in range(10)], rotation=0)

    plt.tight_layout()
    plt.show()


def display_total_errors_barplot(cm, title="Total Misclassifications per Digit"):
    total_instances = np.sum(cm, axis=1)
    correct_guesses = np.diagonal(cm)
    total_errors = total_instances - correct_guesses
    digits = np.arange(10)

    average_errors = np.mean(total_errors)
    avg_label = f"Average: {average_errors:.2f}"

    mnist_display.display_barplot(
        x_vals=digits,
        y_vals=total_errors,
        title=title,
        x_label='Digit Class',
        y_label='Total Number of Errors',
        text_fmt=".0f",
        hline_val=average_errors,
        hline_label=avg_label
    )