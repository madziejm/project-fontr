from typing import Any

import numpy as np
import pandas as pd


def get_label2idx_mapping(idx2label: pd.DataFrame) -> dict:
    labels = idx2label.iloc[:, 0].unique()
    label2idx = {l: i for i, l in enumerate(sorted(labels))}
    return label2idx


def labeled_images_split(
    data: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split labeled image list to train, validation and test dataset

    Args:
        data (pd.DataFrame): list of images
        parameters (dict[str, Any]): pipeline parameters

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: split dataset
    """
    # TODO in-class balance in the split here
    train_frac: float = parameters["data_split"]["train_frac"]
    valid_frac: float = parameters["data_split"]["valid_frac"]
    test_frac: float = parameters["data_split"]["test_frac"]

    assert np.isclose(1.0, train_frac + valid_frac + test_frac)

    train, valid, test = np.split(
        data.sample(
            frac=1,
            random_state=parameters["random_state_seed"],
        ),
        np.array(
            [int(train_frac * len(data)), int((train_frac + valid_frac) * len(data))]
        ),
    )

    return train, valid, test  # type: ignore


def unlabeled_images_split(
    data: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split unlabeled image list to train and test dataset

    Args:
        data (pd.DataFrame): list of images
        parameters (dict[str, Any]): pipeline parameters

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: split dataset
    """

    train_frac: float = parameters["data_split"]["train_frac"]
    valid_frac: float = parameters["data_split"]["valid_frac"]
    test_frac: float = parameters["data_split"]["test_frac"]

    assert np.isclose(1.0, train_frac + valid_frac + test_frac)

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    # (mypy does not know that the returned np.ndarrays are in fact pd.dataframes)
    train, valid, test = np.split(  # type: ignore
        data.sample(
            frac=1,
            random_state=parameters["random_state_seed"],
        ),
        np.array(
            [int(train_frac * len(data)), int((train_frac + valid_frac) * len(data))]
        ),
    )

    return train, valid, test
