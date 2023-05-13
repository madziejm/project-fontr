from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    read_bcf_metadata,
    read_labels,
    upload_bcf_as_png,
    upload_labels_as_csv,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # Real test dataset
            node(
                func=read_bcf_metadata,
                inputs="VFR_real_test_bcf",
                outputs=["real_test_file_pointer", "real_test_sizes"],
                name="real_test_read_bcf_metadata",
            ),
            node(
                func=upload_bcf_as_png,
                inputs=[
                    "VFR_real_test_bcf",
                    "real_test_sizes",
                    "params:VFR_real_test_s3",
                ],
                outputs=None,
                name="real_test_upload_bcf",
            ),
            node(
                func=read_labels,
                inputs="VFR_real_test_label",
                outputs="real_test_label_df",
                name="real_test_read_label",
            ),
            node(
                func=upload_labels_as_csv,
                inputs=["real_test_label_df", "params:VFR_real_test_s3"],
                outputs=None,
                name="real_test_upload_label",
            ),
            # Syn train dataset
            node(
                func=read_bcf_metadata,
                inputs="VFR_syn_train_bcf",
                outputs=["syn_train_file_pointer", "syn_train_sizes"],
                name="syn_train_read_bcf_metadata",
            ),
            node(
                func=upload_bcf_as_png,
                inputs=[
                    "VFR_syn_train_bcf",
                    "syn_train_sizes",
                    "params:VFR_syn_train_s3",
                ],
                outputs=None,
                name="syn_train_upload_bcf",
            ),
            node(
                func=read_labels,
                inputs="VFR_syn_train_label",
                outputs="syn_train_label_df",
                name="syn_train_read_label",
            ),
            node(
                func=upload_labels_as_csv,
                inputs=["syn_train_label_df", "params:VFR_syn_train_s3"],
                outputs=None,
                name="syn_train_upload_label",
            ),
        ]
    )
