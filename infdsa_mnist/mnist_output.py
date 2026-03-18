import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def display_images(x, y, title=None):
    amount = len(x)

    plt.figure(figsize=(3 * amount, 4))
    for i in range(amount):
        plt.subplot(1, amount, i + 1)
        plt.imshow(x[i], cmap='gray')
        plt.title(f"Label: {y[i]}", fontsize=18)
        plt.axis('off')

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def display_grid(x, y, rows, cols, title=None):
    plt.figure(figsize=(3 * cols, 3 * rows))

    for i in range(rows * cols):
        if i >= len(x):
            break

        plt.subplot(rows, cols, i + 1)
        plt.imshow(x[i], cmap='gray')
        plt.title(str(y[i]))
        plt.axis('off')

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def display_barplot(x_vals, y_vals, title, x_label, y_label, palette='Oranges', text_fmt=".0f",
                         hline_val=None, hline_label=None):

    plt.figure(figsize=(10, 5))
    plt.grid(axis='x', visible=False)
    plt.grid(axis='y', color='dimgray', linestyle='-', linewidth=0.8, alpha=0.7, zorder=2)

    ax = sns.barplot(x=x_vals, y=y_vals, hue=y_vals, palette=palette,
                     legend=False, edgecolor='black', zorder=3)
    patches = ax.patches

    plt.title(title, fontsize=14)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_vals)

    max_y = max(y_vals)
    ceiling = max(max_y, hline_val) if hline_val is not None else max_y
    plt.ylim(0, ceiling * 1.2)

    offset = max_y * 0.02

    for bar in patches:
        yval = bar.get_height()
        if np.isnan(yval): continue

        text_str = f"{yval:{text_fmt}}"

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + offset,
            text_str,
            ha='center',
            va='bottom',
            zorder=5,
            color="snow",
            fontweight='bold'
        )

    if hline_val is not None:
        plt.axhline(y=hline_val, color='crimson', linestyle=':', linewidth=2, zorder=4,
                    label=hline_label)
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def create_dataset_table(
    columns: list,
    data: list,
    caption: str = "MNIST Dataset Exploration",
):
    df = pd.DataFrame(data, columns=columns)

    return df.style.set_caption(caption).hide(axis="index")
