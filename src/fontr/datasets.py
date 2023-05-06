from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Optional

import fsspec
import pandas as pd
import torch
import torchvision
from fsspec.spec import AbstractFileSystem
from kedro.io import AbstractDataSet, AbstractVersionedDataSet, DataSetError
from kedro.io.core import get_protocol_and_path
from PIL.Image import Image
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
        path: str,
        path_column: str = "path",
        label_column: str = "label",
        data_file_name: str = "data.csv",
        fs_args: Optional[Dict] = None,
        credentials: Optional[Dict] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        # super(Dataset, self).__init__() # todo
        # super(AbstractDataSet, self).__init__() # todo
        self.target_transform_fn = target_transform
        self.transform_fn = transform
        self.dir_path = path.rstrip("/")
        self.data_file_name = data_file_name

        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}

        protocol, path = get_protocol_and_path(path)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._storage_options = {**_credentials, **_fs_args}

        self._fs: AbstractFileSystem = fsspec.filesystem(
            self._protocol, **self._storage_options
        )
        self.label_column = label_column
        self.path_column = path_column
        self.data: Optional[pd.DataFrame] = None

    def _load(self) -> "KedroPytorchImageDataset":
        self.data = pd.read_csv(self.dir_path + "/data.csv")
        return self

    def _save(self, data: pd.DataFrame):
        self.data = data
        with self._fs.open(self._data_path, "wt", encoding="utf-8") as f:
            data.to_csv(f, index=False)

    @property
    def _data_path(self):
        return Path(self.dir_path) / self.data_file_name

    def _describe(self) -> Dict[str, Any]:
        return {
            "directory": self.dir_path,
            "data_path": self._data_path,
            "num_examples": self.data.shape[0] if self.data else None,
            "status": "initialized" if self.data is not None else "unitialized",
        }

    def transform(self, img: Image) -> torch.Tensor:
        if self.transform_fn:
            return self.transform_fn(img)
        else:
            return torchvision.F.to_tensor(img)

    def target_transform(self, label: Any) -> torch.Tensor:
        if self.target_transform_fn:
            return self.target_transform_fn(label)
        else:
            return torch.tensor(int(label))

    def __getitem__(self, index):
        assert index < self.data.shape[0], "sample index larger than sample count"
        label = self.data.loc[index, self.label_column]
        with self._fs.open(self.data.loc[index, self.path_column], "rb") as f:
            return self.transform(
                Image.open(f).convert("RGB"), self.target_transform(label)
            )

    def __len__(self):
        return self.data.shape[0] if self.data is not None else 0

    def with_transforms(
        self, transform=None, target_transform=None
    ) -> "KedroPytorchImageDataset":
        self.transform_fn = transform
        self.target_transform_fn = target_transform
        return self


class TorchScriptModelDataset(AbstractDataSet):
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
            return {"type": "Torch Script Model"}


class GoogleDriveDataset(AbstractVersionedDataSet):
    def __init__(self, filepath: str, file_name: str):
        protocol, path = get_protocol_and_path(filepath)
        file_id = path.split("/")[-1]

        self.file_id = file_id
        self._protocol = protocol

        self._fs = fsspec.filesystem(self._protocol, **{"root_file_id": file_id})
        super().__init__(
            filepath=PurePosixPath(path),
            version=None,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

        self.file_name = file_name

    def _load(self) -> fsspec.core.OpenFile:
        return self._fs.open(self.file_name, "rb")

    def _save(self, data: Any) -> None:
        raise DataSetError("Read-only dataset")

    def _describe(self) -> Dict[str, Any]:
        return {"fileid": self.file_id}
