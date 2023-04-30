from pathlib import Path
from typing import Dict, Any
from kedro.io import AbstractDataSet, DataSetError


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
