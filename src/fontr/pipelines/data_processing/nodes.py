from typing import Any, Callable

import pandas as pd


def list_files(
    partitioned_file_list: dict[str, Callable[[], Any]],
    parameters: dict,
    limit: int = -1,
) -> pd.DataFrame:
    results = []

    for partition_key, partition_load_func in sorted(partitioned_file_list.items()):
        file_path = partition_load_func()
        results.append(file_path)

    df = pd.DataFrame(results)
    return (
        df if limit < 0 else df.sample(n=limit, random_state=parameters["random_state"])
    )


def get_label2index_mapping(data: pd.DataFrame, parameters: dict) -> dict:
    labels = data[parameters["target_column"]].unique()
    label2index = {l: i for i, l in enumerate(sorted(labels))}
    return label2index


def split_data(
    data: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_train = data.groupby(
        parameters["target_column"], group_keys=True
    ).apply(  # TODO: Perhaps something else
        lambda g: g.sample(
            frac=parameters["train_fraction"],
            random_state=parameters["random_state"],
        )
    )

    data_val = data.drop(data_train.index)
    return data_train, data_val
