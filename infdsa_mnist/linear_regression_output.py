import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def display_metrics(cm, accuracy, models):
    print(f"===== Linear Regression Metrics =====")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Model size: {(len(pickle.dumps(models)) / 1024):.2f} KB")
    print("=====================================")


def display_full_confusion_matrix(cm, title="Linear Regression Confusion Matrix"):
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
