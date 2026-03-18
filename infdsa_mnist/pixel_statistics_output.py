import numpy as np
import matplotlib.pyplot as plt
import infdsa_mnist.mnist_output as mnist_display


def plot_average_pixel_barplot(digit_averages_dict, overall_average=None):
    x_vals = list(digit_averages_dict.keys())
    y_vals = list(digit_averages_dict.values())

    hline_lbl = f'Overall Avg: {overall_average:.2f}' if overall_average else None

    mnist_display.display_barplot(
        x_vals=x_vals,
        y_vals=y_vals,
        title='Average Pixel Value per Digit Class',
        x_label='Digit Class',
        y_label='Average Pixel Intensity (0-255)',
        text_fmt=".2f",
        hline_val=overall_average,
        hline_label=hline_lbl
    )


def plot_average_digit_heatmaps(average_images_dict, interpolation_method="none"):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), layout='constrained')
    fig.suptitle('Normalized Average Pixel Values per Digit Class', fontsize=16)

    im = None

    for digit, ax in enumerate(axes.flatten()):
        img_array = average_images_dict[digit].reshape(28, 28)

        img_normalized = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))

        # im = ax.imshow(img_normalized, cmap='plasma', interpolation=interpolation_method)
        im = ax.imshow(img_normalized, cmap='gray', interpolation=interpolation_method)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(f'Digit: {digit}')

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.05)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Least', 'Average', 'Most'])

    plt.show()


def plot_standard_deviation_heatmap(std_images_dict, interpolation_method="none"):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), layout='constrained')
    fig.suptitle('Standard Deviation of Pixel Values per Digit Class', fontsize=16)

    im = None

    for digit, ax in enumerate(axes.flatten()):
        img_array = std_images_dict[digit].reshape(28, 28)

        img_normalized = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))

        im = ax.imshow(img_normalized, cmap='gray', interpolation=interpolation_method)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(f'Digit: {digit}')

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.05)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Least', 'Average', 'Most'])

    plt.show()


def plot_average_pixel_value_across_dataset_heatmap(average_pixel_values, interpolation_method="none"):
    fig, ax = plt.subplots(figsize=(14, 14))
    img_data = average_pixel_values.reshape(28, 28)

    ax.imshow(img_data, cmap='gray', interpolation=interpolation_method)

    threshold = img_data.max() / 2.

    for i in range(28):
        for j in range(28):
            text_color = "black" if img_data[i, j] > threshold else "white"
            ax.text(j, i, f"{img_data[i, j]:.2f}",
                    ha="center", va="center", color=text_color, fontsize=10)

    ax.set_title('Average Pixel Value Across the Dataset')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_standard_deviation_pixel_value_across_dataset_heatmap(standard_deviation_pixel_values, interpolation_method="none"):
    fig, ax = plt.subplots(figsize=(14, 14))
    img_data = standard_deviation_pixel_values.reshape(28, 28)

    ax.imshow(img_data, cmap='gray', interpolation=interpolation_method)

    threshold = img_data.max() / 2.

    for i in range(28):
        for j in range(28):
            text_color = "black" if img_data[i, j] > threshold else "white"
            ax.text(j, i, f"{img_data[i, j]:.2f}",
                    ha="center", va="center", color=text_color, fontsize=10)

    ax.set_title('Standard Deviation Pixel Value Across the Dataset')
    plt.axis('off')

    plt.tight_layout()
    plt.show()