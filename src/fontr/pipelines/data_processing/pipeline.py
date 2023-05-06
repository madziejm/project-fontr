from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_label2index_mapping, list_files, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=list_files,
                inputs=["train_images_list"],
                outputs="train_data_all",
                name="list_train_files",
            ),
            node(
                func=get_label2index_mapping,
                inputs=["train_data_all", "paremeters"],
                outputs="label2index",
                name="get_label2index",
            ),
            node(
                func=split_data,
                inputs=["train_data_all", "parameters"],
                outputs=["train_dataset", "val_dataset"],
                name="split_train_validation",
            ),
        ]
    )
