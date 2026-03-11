import pandas as pd
from .mnist_display import create_dataset_table


def create_image_count_df(train_images, test_images, total_images):
    return create_dataset_table(
        columns=["Set", "Images"],
        data=[["Training", train_images], ["Testing", test_images], ["Total", total_images]],
        caption="Number of Images in the MNIST Dataset"
    )


def create_digit_distribution(distribution):
    return create_dataset_table(
        columns=["Digit", "Count", "Percentage"],
        data=[[digit, count, f"{count / sum(distribution.values()) * 100:.2f}%"] for digit, count in distribution.items()],
        caption="Distribution of Digits in the MNIST Dataset"
    )


def create_most_least_df(distribution):
    dist_series = pd.Series(distribution)

    return create_dataset_table(
        columns=["Category", "Digit", "Count"],
        data=[
            ["Most Common", dist_series.idxmax(), dist_series.max()],
            ["Least Common", dist_series.idxmin(), dist_series.min()]
        ],
        caption="Most and Least Common Digits"
    )


def create_image_info_df(shape, dtype):
    return create_dataset_table(
        columns=["Attribute", "Value"],
        data=[["Shape", shape], ["Datatype", dtype]],
        caption="Shape and Datatype of a Single Image in the MNIST Dataset")


def create_memory_usage_df(single_image_memory, total_memory):
    return create_dataset_table(
        columns=["Attribute", "Value"],
        data=[
            ["Single Image Memory Usage", f"{single_image_memory} B"],
            ["Total Memory Usage", f"{total_memory / (1024 ** 2):.2f} MB"]
        ],
        caption="Memory Usage of a Single Image and the Entire Dataset"
    )


def create_dataset_summary_df(shape, dtype, train_count, test_count, total_count, balance, user_threshold):
    return create_dataset_table(
        columns=["Attribute", "Value"],
        data=[
            ["Shape of a Single Image", shape],
            ["Datatype of a Single Image", dtype],
            ["Number of Training Images", train_count],
            ["Number of Testing Images", test_count],
            ["Total Number of Images", total_count],
            ["Balance Threshold", f"{user_threshold:.2f}"],
            ["Balanced Status", balance]
        ],
        caption="Summary of MNIST Dataset Exploration"
    )


