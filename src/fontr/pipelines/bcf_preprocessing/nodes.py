import fsspec
import numpy as np
import pandas as pd
from kedro.io.core import get_protocol_and_path


def read_bcf_metadata(
    bcf_file: fsspec.core.OpenFile,
) -> tuple[fsspec.core.OpenFile, np.ndarray]:
    """
    Reads metadata of bcf file using the potiner given as the argument.

    The .bcf format looks as follows:
        * 8 bytes - n number of the .png files in the .bcf file.
        * 8n      - size of each .png file.
        * n       - .png files stored as raw bytes.

    Args:
        bcf_file (fsspec.core.OpenFile): File descriptior to the .bcf file.

    Returns:
        tuple[fsspec.core.OpenFile, np.ndarray]:
            File descriptor to the .bcf file.
            Read sizes of the .png files.
    """
    size = int(np.frombuffer(bcf_file.read(8), dtype=np.uint64)[0])
    file_sizes = np.frombuffer(bcf_file.read(8 * size), dtype=np.uint64)

    return bcf_file, file_sizes


def upload_bcf_as_png(
    bcf_file: fsspec.core.OpenFile, file_sizes: np.ndarray, output_path: str
) -> None:
    """
    Stores .png files stored in a .bcf files in a `output_path`.

    Args:
        bcf_file (fsspec.core.OpenFile): File descriptior to the .bcf file.
        file_sizes (np.ndarray): File sizes read in `read_bcf_metadata` node.
        output_path (str): Path where the .png files are stored
    """
    offsets = np.append(np.uint64(0), np.add.accumulate(file_sizes))

    protocol, _ = get_protocol_and_path(output_path)
    _fs = fsspec.filesystem(protocol)

    for i in range(len(file_sizes)):
        bcf_file.seek(np.uint64(len(offsets) * 8 + offsets[i]))
        out = bcf_file.read(offsets[i + 1] - offsets[i])

        filename = f"{output_path}/{i}.png"
        with _fs.open(filename, "wb") as f:
            f.write(out)

    return None


def read_labels(label_file: fsspec.core.OpenFile) -> pd.DataFrame:
    """
    Stores reads labels saved under `label_file` and converts it
    into a cvs file

    Args:
        label_file (fsspec.core.OpenFile): File descriptor to the .label file

    Returns:
        pd.DataFrame: Read labels as dataframe
    """
    labels = np.frombuffer(label_file.read(), dtype=np.uint32)
    df_labels = pd.DataFrame(data=labels, columns=["labels"])
    return df_labels


def upload_labels_as_csv(df_labels: pd.DataFrame, output_path: str):
    """
    Stores passed `df_labels` in a `output_path`.

    Args:
        df_labels (pd.DataFrame): labels
        output_path (str): Pathe where the labels.csv file is stored
    """
    df_labels.to_csv(f"{output_path}/labels.csv")
    return None
