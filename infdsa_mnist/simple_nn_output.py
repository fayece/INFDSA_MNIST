import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_nn_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    early_stopping = getattr(history, 'early_stopping', None)
    best_epoch = len(acc)
    if early_stopping and early_stopping.stopped_epoch > 0:
        best_epoch = early_stopping.stopped_epoch - early_stopping.patience + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    train_acc_best_percent = acc[best_epoch - 1] * 100
    val_acc_best_percent = val_acc[best_epoch - 1] * 100

    ax1.plot(epochs, acc, color='tab:blue', linewidth=2, label=f'Training ({train_acc_best_percent:.2f}% at Epoch {best_epoch})')
    ax1.plot(epochs, val_acc, color='tab:orange', linewidth=2, label=f'Validation ({val_acc_best_percent:.2f}% at Epoch {best_epoch})')
    ax1.axvline(x=best_epoch, color='crimson', linestyle=':', linewidth=2, label=f'Best Weights (Epoch {best_epoch})')
    ax1.set_title('Accuracy', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend()

    ax2.plot(epochs, loss, color='tab:blue', linewidth=2, label=f'Training ({loss[best_epoch - 1]:.4f} at Epoch {best_epoch})')
    ax2.plot(epochs, val_loss, color='tab:orange', linewidth=2, label=f'Validation ({val_loss[best_epoch - 1]:.4f} at Epoch {best_epoch})')
    ax2.axvline(x=best_epoch, color='crimson', linestyle=':', linewidth=2, label=f'Best Weights (Epoch {best_epoch})')
    ax2.set_title('Loss', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def nn_summary(model_or_filename, filename=None):
    if isinstance(model_or_filename, str):
        filename = model_or_filename
        filepath = f"../models/{filename}"

        size_bytes = os.path.getsize(filepath)
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024

        print(f"File: {filename}")
        print(f"Size (KB): {size_kb:.2f} KB")
        print(f"Size (MB): {size_mb:.2f} MB")

    else:
        model = model_or_filename
        filepath = f"../models/{filename}"

        size_bytes = os.path.getsize(filepath)
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024

        print(f"Model: {filename}")
        print(f"Size (KB): {size_kb:.2f} KB")
        print(f"Size (MB): {size_mb:.2f} MB")


def display_full_confusion_matrix(cm, title="Simple Neural Network Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        cbar=True,
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )

    plt.title(title, fontsize=18, pad=20)
    plt.xlabel("Predicted Digit", fontsize=14, labelpad=10)
    plt.ylabel("Actual Digit Label", fontsize=14, labelpad=10)

    ticks = [i + 0.5 for i in range(10)]
    labels = [str(i) for i in range(10)]
    plt.xticks(ticks=ticks, labels=labels)
    plt.yticks(ticks=ticks, labels=labels, rotation=0)

    plt.tight_layout()
    plt.show()


def plot_model_comparison(full_acc, quant_acc, full_path, opt_path, quant_path, storage_threshold_kb= 1024, ram_threshold_kb= 256):
    full_kb = os.path.getsize(full_path) / 1024
    opt_kb = os.path.getsize(opt_path) / 1024
    quant_kb = os.path.getsize(quant_path) / 1024

    fig, ax1 = plt.subplots(figsize=(10, 6))

    categories = ['Full Model\n(W/ Optimizer)', 'Optimized Model\n(No Optimizer)', 'Quantized Model\n(INT8 TFLite)']
    size_values = [full_kb, opt_kb, quant_kb]
    acc_values = [full_acc * 100, full_acc * 100, quant_acc * 100]
    colors = ['tab:red', 'tab:orange', 'tab:green']

    bars = ax1.bar(categories, size_values, color=colors, alpha=0.8, width=0.5)

    ax1.axhline(y=storage_threshold_kb, color='crimson', linestyle='--', linewidth=2,
                label=f'{storage_threshold_kb} KB Storage Limit')
    ax1.axhline(y=ram_threshold_kb, color='gold', linestyle='-.', linewidth=2, label=f'{ram_threshold_kb} KB RAM Limit')

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:,.2f} KB',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold', color='white')

    ax1.set_ylabel('Size (KB)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(size_values) * 1.25)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(categories, acc_values, color='white', marker='o', linestyle='-', linewidth=3, markersize=10,
             label='Test Accuracy')

    for i, acc_val in enumerate(acc_values):
        ax2.annotate(f'{acc_val:.2f}%',
                     xy=(i, acc_val),
                     xytext=(0, 15),
                     textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold', color='white')

    min_acc = min(acc_values)
    ax2.set_ylim(min_acc - 2, 100.5)
    ax2.set_ylabel('Accuracy (%)', color='white', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='white')
    ax2.legend(loc='upper right')
    ax2.grid(False)

    plt.title('Final Review: Model Size vs. Accuracy Trade-off', fontsize=16, pad=15)
    fig.tight_layout()
    plt.show()
