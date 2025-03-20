from pathlib import Path
from typing import Final
import zipfile

import pandas as pd

__all__ = ["load_data"]

_DATA_FOLDER: Final[Path] = Path(__file__).parent.parent / "data"


def extract_data() -> None:
    zip_file: Path = _DATA_FOLDER / "archive.zip"
    with zipfile.ZipFile(zip_file) as archive:
        for file in archive.namelist():
            archive.extract(member=file, path=zip_file.parent)


def load_data() -> pd.DataFrame:
    return pd.read_csv(
        _DATA_FOLDER / "Bank Customer Churn Prediction.csv",
        sep=",",
        index_col=None
    )


from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(strategy="quantile", n_bins=10, encode="ordinal")