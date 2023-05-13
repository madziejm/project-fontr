from kedro.pipeline import Pipeline, pipeline

# from .nodes import (
#     evaluate_classifier,
#     train_pytorch_autoencoder,
#     train_pytorch_classifier
# )  # todo readd


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])
    # TODO: Add autoencoder and classifier training and evaluation
    # return pipeline(
    # [
    #     node(
    #         func=split_data,
    #         inputs=["model_input_table", "params:model_options"],
    #         outputs=["X_train", "X_test", "y_train", "y_test"],
    #         name="split_data_node",
    #     ),
    #     node(
    #         func=train_pytorch_autoencoder,
    #         inputs=["X_train", "y_train"],
    #         outputs="regressor",
    #         name="train_model_node",
    #     ),
    #     node(
    #         func=evaluate_model,
    #         inputs=["regressor", "X_test", "y_test"],
    #         outputs=None,
    # c        name="evaluate_model_node",
    #     ),
    # ]
    # )
