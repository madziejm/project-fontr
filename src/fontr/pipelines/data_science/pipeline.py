from kedro.pipeline import Pipeline, node, pipeline

from fontr.pipelines.nodes import set_random_state

from .nodes import (  # noqa: F401 # TODO
    evaluate_autoencoder,
    evaluate_classifier,
    serialize_model_to_torch_jit,
    train_autoencoder,
    train_classifier,
    predict,
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
                func=train_autoencoder,
                inputs=["syn_train@torch", "syn_valid@torch", "params:autoencoder"],
                outputs="autoencoder",
                name="train_autoencoder_node",
            ),
            # the evaluation is not implemented, skip for now TODO
            # node(
            #     func=evaluate_autoencoder,
            #     inputs=["autoencoder", "syn_train@torch"],
            #     outputs=None,
            #     name="evaluate_autoencoder_node",
            # ),
            node(
                func=train_classifier,
                inputs=[
                    "real_train@torch",
                    "real_valid@torch",
                    "label2idx",
                    "params:classifier",
                    "autoencoder",
                ],
                outputs="classifier",
                name="train_classifier_node",
            ),
            node(
                func=serialize_model_to_torch_jit,
                inputs=[
                    "classifier",
                    "params:classifier.torch_jit_serialization_method",
                ],
                outputs="classifier_torchscript",
                name="serialize_classifier_node_to_jit",
            ),
            # the evaluation is not implemented, skip for now TODO
            # node(
            #     func=evaluate_classifier,
            #     inputs=["classifier", "real_test@torch"],
            #     outputs=None,
            #     name="evaluate_classifier_node",
            # ),
            node(
                func=predict,
                inputs=[
                    "classifier",
                    "label2idx",
                ],
                outputs="predict_output",
                name="predict_node",
            ),
        ]
    )
