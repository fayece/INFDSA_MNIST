import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


def plot_depth_experiment(depths, accuracies, all_predictions, actual_labels):
    num_samples = len(actual_labels)
    num_depths = len(depths)

    row_labels = [f"Img {i + 1} (Actual: {actual_labels[i]})" for i in range(num_samples)]

    table_data = {}
    correct_counts_per_image = [0] * num_samples

    for i, d in enumerate(depths):
        col_name = f"Depth {d}"
        preds = all_predictions[col_name]

        formatted_column = []
        for j in range(num_samples):
            guess = preds[j]
            actual = actual_labels[j]

            if guess == actual:
                formatted_column.append(f"RIGHT ({guess})")
                correct_counts_per_image[j] += 1
            else:
                formatted_column.append(f"WRONG ({guess})")

        table_data[col_name] = formatted_column

    image_acc_col = [f"{int((count / num_depths) * 100)}%" for count in correct_counts_per_image]
    table_data["Image Accuracy"] = image_acc_col

    detailed_df = pd.DataFrame(table_data, index=row_labels)

    accuracy_row = [f"{acc}%" for acc in accuracies] + ["-"]
    detailed_df.loc["Overall Accuracy"] = accuracy_row

    display(detailed_df)

    plt.figure(figsize=(8, 5))
    plt.plot(depths, accuracies, marker='o', linestyle='-', color='orange', linewidth=2)

    plt.title("Decision Tree Accuracy vs. Max Depth", fontsize=14)
    plt.xlabel("Maximum Tree Depth", fontsize=12)
    plt.ylabel("Accuracy on Unseen Data (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(depths)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()


def plot_systems_report(metrics: dict):
    table_data = [
        ["RAM", "Dataset Footprint", f"{metrics['dataset_ram_kb']:.2f} KB", "-", "-"],
        ["RAM", f"Tree Footprint ({metrics['total_nodes']} nodes)", f"{metrics['tree_ram_kb']:.2f} KB", "-", "-"],
        ["RAM", "Total Peak RAM", f"{metrics['total_ram_kb']:.2f} KB", "256.00 KB",
         "SUCCESS" if metrics['total_ram_kb'] <= 256 else "FAILED"],

        ["Storage", "Serialized Model", f"{metrics['tree_storage_kb']:.2f} KB", "-", "-"],
        ["Storage", "Dataset Binary", f"{metrics['dataset_storage_kb']:.2f} KB", "-", "-"],
        ["Storage", "Total Disk Usage", f"{metrics['total_storage_kb']:.2f} KB", "1024.00 KB",
         "SUCCESS" if metrics['total_storage_kb'] <= 1024 else "FAILED"],

        ["Hardware", "Compute Targeted", "CPU Only (NumPy, built-ins)", "No GPU", "SUCCESS"]
    ]

    df = pd.DataFrame(table_data, columns=["Category", "Metric", "Usage", "Limit", "Status"])

    styled_report = (
        df.style
        .set_caption("MysteryDevice System Validation Report")
        .hide(axis="index")
    )

    display(styled_report)