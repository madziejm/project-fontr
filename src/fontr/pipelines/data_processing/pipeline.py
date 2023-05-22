from kedro.pipeline import Pipeline, node, pipeline  # noqa: F401

from fontr.pipelines.nodes import set_random_state

from .nodes import (  # noqa: F401
    get_label2idx_mapping,
    labeled_images_split,
    unlabeled_images_split,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # hmm this would work with SequentialRunner only I guess
            node(
                func=set_random_state,
                inputs=("params:random_state_seed"),
                outputs=None,
                name="set_random_state_node",
            ),
            node(
                func=get_label2idx_mapping,
                inputs=["idx2label"],
                outputs="label2idx",
                name="get_label2idx_node",
            ),
            node(
                func=labeled_images_split,
                inputs=["real_dataset", "parameters"],
                outputs=["real_train@csv", "real_valid@csv", "real_test@csv"],
                name="split_train_validation",
            ),
            node(
                func=unlabeled_images_split,
                inputs=["syn_dataset", "parameters"],
                outputs=["syn_train@csv", "syn_valid@csv", "syn_test@csv"],
                name="unlabeled_images_split_node",
            ),
        ]
    )
