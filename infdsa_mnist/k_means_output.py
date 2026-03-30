import matplotlib.pyplot as plt
import seaborn as sns
from infdsa_mnist.mnist_output import display_grid, display_total_errors_barplot

def plot_prototypes(prototypes, labels, k_val=10):
    display_grid(
        x=prototypes,
        y=labels,
        rows=10,
        cols=k_val,
        title=f'K-Means Prototypes (k={k_val})')

def plot_prediction(unseen_digit, prediction):
    display_grid(
        x=[unseen_digit],
        y=[prediction],
        rows=1,
        cols=1,
        title=f'K-Means Prediction: {prediction}')


def display_full_confusion_matrix(cm, title="K-Means Confusion Matrix"):
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
    plt.xlabel("Predicted Cluster/Digit", fontsize=14, labelpad=10)
    plt.ylabel("Actual Digit Label", fontsize=14, labelpad=10)

    ticks = [i + 0.5 for i in range(10)]
    labels = [str(i) for i in range(10)]
    plt.xticks(ticks=ticks, labels=labels)
    plt.yticks(ticks=ticks, labels=labels, rotation=0)

    plt.tight_layout()
    plt.show()


def display_evaluation_metrics(eval_results):
    k = eval_results['k']
    acc = eval_results['accuracy']
    mem = eval_results['memory_kb']
    t = eval_results['time_seconds']

    print(f"========== K-Means Evaluation (k={k}) ==========")
    print(f"Accuracy:      {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Memory Usage:  {mem:,} KB")
    print(f"Execution Time:{t:.2f} seconds")
    print("================================================")

    cm = eval_results['confusion_matrix']

    display_total_errors_barplot(cm, title=f"Total Misclassifications per Digit (k={k})")

def plot_accuracy_vs_memory(results):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Prototypes per Digit (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', color=color1, fontsize=12, fontweight='bold')
    ax1.plot(results['k'], results['accuracy'], marker='o', color=color1, linewidth=2, markersize=8, label="Accuracy")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Memory Usage (KB)', color=color2, fontsize=12, fontweight='bold')
    ax2.plot(results['k'], results['memory_kb'], marker='s', linestyle='--', color=color2, linewidth=2, markersize=8, label="Memory")
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('K-Means Performance: Accuracy vs. Memory Usage', fontsize=16, pad=15)
    fig.tight_layout()
    plt.show()

def plot_memory_compression(raw_kb, storage_kb, peak_kb, k_val, accuracy, threshold=256):
    fig, ax = plt.subplots(figsize=(9, 6))

    categories = ['Original (float64)', 'Binned Storage (2-bit)', 'Peak Execution (int32)']
    values = [raw_kb, storage_kb, peak_kb]
    colors = ['tab:red', 'tab:green', 'tab:blue']

    bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.5)

    ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'{threshold} KB Hardware Limit')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,.2f} KB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')

    plt.title(f'Post-Training Compression Impact (k={k_val})\nAchieved Accuracy: {accuracy * 100:.2f}%',
              fontsize=16, pad=15)

    ax.set_ylim(0, max(values) * 1.15)

    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()