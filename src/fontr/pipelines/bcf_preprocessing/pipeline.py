from kedro.pipeline import Pipeline, node, pipeline

from .nodes import read_bcf_metadata, upload_bcf_as_png


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=read_bcf_metadata,
                inputs="syn_train_bcf",
                outputs=["syn_train_file_pointer", "syn_train_sizes"],
                name="read_bcf_metadata",
            ),
            node(
                func=upload_bcf_as_png,
                inputs=[
                    "syn_train_file_pointer",
                    "syn_train_sizes",
                    "params:output_path",
                ],
                outputs=None,
                name="upload_bcf",
            ),
        ]
    )
