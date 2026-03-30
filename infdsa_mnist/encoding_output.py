from infdsa_mnist import mnist_output


def plot_encoding_comparison(
        results: dict,
        title_prefix: str = "",
        baseline_accuracy: float = None,
        baseline_bytes: float = None
):
    techniques = list(results.keys())
    accuracies = [results[t]["accuracy"] * 100 for t in techniques]
    avg_bytes = [results[t]["avg_bytes"] for t in techniques]

    prefix = f"{title_prefix} " if title_prefix else ""

    ref_accuracy = baseline_accuracy if baseline_accuracy is not None else accuracies[0]
    ref_bytes = baseline_bytes if baseline_bytes is not None else avg_bytes[0]

    mnist_output.display_barplot(
        x_vals=techniques,
        y_vals=accuracies,
        title=f"{prefix}Encoding Technique Comparison (Higher is Better)",
        x_label="Encoding Technique",
        y_label="Accuracy (%)",
        text_fmt=".2f",
        hline_val=ref_accuracy,
        hline_label=f"Baseline: {ref_accuracy:.2f}%",
    )

    mnist_output.display_barplot(
        x_vals=techniques,
        y_vals=avg_bytes,
        title=f"{prefix}Average Memory per Image (Lower is Better)",
        x_label="Encoding Technique",
        y_label="Bytes",
        text_fmt=".1f",
        hline_val=ref_bytes,
        hline_label=f"Baseline: {ref_bytes:.1f} B"
    )

# def plot_encoding_comparison(results: dict):
#     techniques = list(results.keys())
#     accuracies = [results[t]["accuracy"] * 100 for t in techniques]
#     avg_bytes = [results[t]["avg_bytes"] for t in techniques]
#
#     baseline_accuracy = accuracies[0]
#     baseline_bytes = avg_bytes[0]
#
#     fig = plt.figure(figsize=(14, 6))
#     fig.suptitle("Encoding Technique Comparison", fontsize=14, fontweight="bold", y=1.01)
#     gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
#
#     colors = [
#         "#4C72B0" if acc >= baseline_accuracy else "#DD8452"
#         for acc in accuracies
#     ]
#
#     # --- Left: Accuracy ---
#     ax1 = fig.add_subplot(gs[0])
#     bars = ax1.bar(techniques, accuracies, color=colors, edgecolor="white", linewidth=0.8)
#     ax1.axhline(baseline_accuracy, color="#4C72B0", linestyle="--", linewidth=1.2, label="Baseline")
#     ax1.set_title("Accuracy per Encoding", fontweight="bold")
#     ax1.set_ylabel("Accuracy (%)")
#     ax1.set_ylim(0, 100)
#     ax1.set_xticks(range(len(techniques)))
#     ax1.set_xticklabels(techniques, rotation=25, ha="right", fontsize=9)
#     ax1.legend(fontsize=8)
#     for bar, acc in zip(bars, accuracies):
#         ax1.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height() + 1,
#             f"{acc:.1f}%",
#             ha="center", va="bottom", fontsize=8
#         )
#
#     ax2 = fig.add_subplot(gs[1])
#     mem_colors = [
#         "#4C72B0" if b <= baseline_bytes else "#DD8452"
#         for b in avg_bytes
#     ]
#     bars2 = ax2.bar(techniques, avg_bytes, color=mem_colors, edgecolor="white", linewidth=0.8)
#     ax2.axhline(baseline_bytes, color="#4C72B0", linestyle="--", linewidth=1.2, label="Baseline")
#     ax2.set_title("Avg Memory per Image", fontweight="bold")
#     ax2.set_ylabel("Bytes")
#     ax2.set_xticks(range(len(techniques)))
#     ax2.set_xticklabels(techniques, rotation=25, ha="right", fontsize=9)
#     ax2.legend(fontsize=8)
#     for bar, b in zip(bars2, avg_bytes):
#         ax2.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height() + 0.3,
#             f"{b:.1f}B",
#             ha="center", va="bottom", fontsize=8
#         )
#
#     plt.tight_layout()
#     plt.show()
