from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, Optional

import fsspec
import pandas as pd
import torch
import torchvision
from fsspec.spec import AbstractFileSystem
from kedro.io import AbstractDataSet, AbstractVersionedDataSet, DataSetError
from kedro.io.core import get_protocol_and_path
from PIL import Image
from torch.jit import ScriptModule
from torch.utils.data import Dataset


class FileWithDirAsLabel(AbstractDataSet):
    def __init__(self, filepath: str):
        self.path = filepath

    def _load(self) -> dict:
        p = PurePosixPath(self.path)
        return {"path": self.path, "label": p.parent.name}

    def _save(self, data: Any) -> None:
        raise DataSetError("Read-only dataset")

    def _describe(self) -> Dict[str, Any]:
        return {}


class KedroPytorchImageDataset(Dataset, AbstractDataSet):
    def __init__(
        self,
        filepath: str,
        path_column: int = 0,
        label_column: int = 1,
        fs_args: Optional[Dict] = None,
        credentials: Optional[Dict] = None,
        transform: Optional[Callable] = None,
        transform_copy: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_labels=True,
        add_copies=False,
    ):
        """torch.utils.data.Dataset mixed with kedro.io.AbstractDataSet.
        filepath should be a CSV listing paths of images relative to the directory
        of filepath in the first column. The optional label_column column can
        contain labels for the images. The images should be located in the
        directory mentioned before.
        TODO fix this docstring.
        """
        Dataset.__init__(self)
        AbstractDataSet.__init__(self)
        self.target_transform_fn = target_transform
        self.transform_fn = transform
        self.copy_transform = transform_copy

        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}

        protocol, path = get_protocol_and_path(filepath)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._storage_options = {**_credentials, **_fs_args}

        self._fs: AbstractFileSystem = fsspec.filesystem(
            self._protocol, **self._storage_options
        )

        self.filepath = filepath
        self.dir_path = self._fs._parent(filepath)
        assert (
            self.dir_path[-1] != "/"
        ), "this is stupid, yet we cannot have double trailing slash in string interpolation later on\
        (this could point to a different key in S3)"

        self.label_column = label_column
        self.path_column = path_column
        self.data: Optional[pd.DataFrame] = None

        self.return_labels = return_labels
        self.add_copies = add_copies

    def _load(self) -> "KedroPytorchImageDataset":
        if self.add_copies:
            df = pd.read_csv(self.filepath)
            df['copy'] = 0
            df2 = df.copy(deep=True)
            df2['copy'] = 1
            self.data = pd.concat([df, df2])
        else:
            self.data = pd.read_csv(self.filepath)
        return self

    def _save(self, data: pd.DataFrame):
        self.data = data
        with self._fs.open(self.filepath, "wt", encoding="utf-8") as f:
            data.to_csv(f, index=False)

    def _describe(self) -> Dict[str, Any]:
        return {
            "directory": self.dir_path,
            "filepath": self.filepath,
            "num_examples": self.data.shape[0] if self.data else None,
            "status": "initialized" if self.data is not None else "unitialized",
        }

    def transform(self, img: Image.Image) -> torch.Tensor:
        img = torchvision.transforms.functional.to_tensor(img)
        return self.transform_fn(img) if self.transform_fn else img

    def target_transform(self, label: Any) -> torch.Tensor:
        if self.target_transform_fn:
            return self.target_transform_fn(label)
        else:
            return torch.tensor(int(label))

    def __getitem__(self, index):
        assert self.data is not None
        assert index < self.data.shape[0], "sample index larger than sample count"
        label = self.data.iloc[index, self.label_column]
        img_path = f"{self.dir_path}/{self.data.iloc[index, self.path_column]}.png"
        # I wish we had extensions in the CSV
        with self._fs.open(img_path, "rb") as f:
            if self.add_copies and self.data.iloc[index]['copy']:
                img = self.copy_transform(Image.open(f).convert("RGB"))
            else:
                img = self.transform(Image.open(f).convert("RGB"))
            if self.return_labels:
                return img, self.target_transform(label)
            else:
                return img

    def __len__(self):
        return self.data.shape[0] if self.data is not None else 0

    def with_transforms(
        self, transform=None, target_transform=None, copy_transform=None
    ) -> "KedroPytorchImageDataset":
        self.transform_fn = transform
        self.target_transform_fn = target_transform
        self.copy_transform = copy_transform
        return self


class TorchScriptModelDataset(AbstractDataSet):
    """Kedro DataSet for a model to be (de-)serialized with torch.jit.{load,save}"""

    def __init__(
        self,
        filepath: str,
        map_location: str = "cpu",
        fs_args: Optional[Dict] = None,
        credentials: Optional[Dict] = None,
    ):
        super().__init__()
        self.filepath = filepath
        self.map_location = map_location

        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}  # noqa: F841

        protocol, _ = get_protocol_and_path(filepath)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._storage_options = {**_credentials, **_fs_args}

        self._fs: AbstractFileSystem = fsspec.filesystem(
            self._protocol, **self._storage_options
        )

    def _load(self) -> ScriptModule:
        with self._fs.open(self.filepath, "rb") as f:
            return torch.jit.load(f, self.map_location)

    def _save(self, data: ScriptModule):
        with self._fs.open(self.filepath, "wb") as f:
            return torch.jit.save(data, f)

    def _describe(self) -> Dict[str, Any]:
        return {
            "type": "Torch Script Model",
            "filepath": self.filepath,
            "protocol": self._protocol,
        }


class TorchPickleModelDataset(AbstractDataSet):
    """Kedro DataSet for a model to be (de-)serialized with torch.{load,save}"""

    def __init__(
        self,
        filepath: str,
        map_location: str = "cpu",
        fs_args: Optional[Dict] = None,
        credentials: Optional[Dict] = None,
    ):
        super().__init__()
        self.filepath = filepath
        self.map_location = map_location

        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}  # noqa: F841

        protocol, _ = get_protocol_and_path(filepath)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._storage_options = {**_credentials, **_fs_args}

        self._fs: AbstractFileSystem = fsspec.filesystem(
            self._protocol, **self._storage_options
        )

    def _load(self) -> Any:
        with self._fs.open(self.filepath, "rb") as f:
            return torch.load(f, self.map_location)

    def _save(self, data: Any):
        with self._fs.open(self.filepath, "wb") as f:
            return torch.save(data, f)

    def _describe(self) -> Dict[str, Any]:
        return {
            "type": "Torch Model",
            "filepath": self.filepath,
            "protocol": self._protocol,
        }


class GoogleDriveDataset(AbstractVersionedDataSet):
    def __init__(self, filepath: str, file_name: str):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self.file_id = path.split("/")[-1]

        self.file_name = file_name

        self.__fs: Optional[fsspec.spec.AbstractFileSystem] = None

        super().__init__(
            filepath=PurePosixPath(path),
            version=None,
        )

    @property
    def _fs(self):
        if self.__fs is None:
            self.__fs = fsspec.filesystem(self._protocol, root_file_id=self.file_id)
            self._glob_function = self.__fs.glob
        return self.__fs

    def _load(self) -> fsspec.core.OpenFile:
        return self._fs.open(self.file_name, "rb")

    def _save(self, data: Any) -> None:
        raise DataSetError("Read-only dataset")

    def _describe(self) -> Dict[str, Any]:
        return {"fileid": self.file_id}

    def exists(self) -> bool:
        return self._fs.exists()
