from pathlib import Path
from typing import Dict, Any
from kedro.io import AbstractDataSet, DataSetError, AbstractVersionedDataSet
from kedro.io.core import get_protocol_and_path
import fsspec


class FileWithDirAsLabel(AbstractDataSet):
    def __init__(self, filepath: str):
        self.path = filepath

    def _load(self) -> dict:
        p = Path(self.path)
        return {"path": self.path, "label": p.parent.name}

    def _save(self, data: Any) -> None:
        raise DataSetError("Read-only dataset")

    def _describe(self) -> Dict[str, Any]:
        pass


class GoogleDriveDataset(AbstractVersionedDataSet):
    def __init__(self, filepath: str, file_name: str):
        version = None

        protocol, path = get_protocol_and_path(filepath, version)
        file_id = path.split("/")[-1]

        self.file_id = file_id
        self._protocol = protocol

        self._fs = fsspec.filesystem(self._protocol, **{"root_file_id": file_id})
        super().__init__(
            filepath=Path(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

        self.file_name = file_name

    def _load(self) -> fsspec.core.OpenFile:
        return self._fs.open(self.file_name, "rb")

    def _save(self, data: Any) -> None:
        raise DataSetError("Read-only dataset")

    def _describe(self) -> Dict[str, Any]:
        pass
