# tests/unit/test_data_utils.py

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.utils.data_utils import DataUtils


@patch("src.utils.data_utils.Path.exists", return_value=True)
def test_source_raw_data_from_kaggle_returns_existing_path(mock_exists):
    path = DataUtils.source_raw_data_from_kaggle()
    assert isinstance(path, Path)
    assert "heart.csv" in str(path)


@patch("src.utils.data_utils.pd.read_csv")
@patch("src.utils.data_utils.Path.exists", return_value=True)
def test_load_and_process_data_basic(mock_exists, mock_read_csv):
    # Setup dummy DataFrame
    import pandas as pd
    df = pd.DataFrame({
        "Feature1": [1, 2, 3, 4, 5, 6],
        "Feature2": ["a", "b", "c", "d", "e", "f"],
        "HeartDisease": [0, 1, 0, 1, 0, 1]
    })
    mock_read_csv.return_value = df

    # Act
    X_train, y_train, X_val, y_val, X_test, y_test = DataUtils.load_and_process_data("dummy.csv")

    # Assert
    assert len(X_train) > 0
    assert len(y_test) > 0
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)


@patch("src.utils.data_utils.kagglehub.dataset_download")
@patch("src.utils.data_utils.Path.glob")
@patch("src.utils.data_utils.Path.exists", return_value=False)
@patch("src.utils.data_utils.shutil.copy2")
def test_download_if_not_exists(mock_copy, mock_glob, mock_exists, mock_download):
    mock_glob.return_value = [Path("mock.csv")]
    mock_download.return_value = "download/path"

    result = DataUtils.source_raw_data_from_kaggle()
    assert isinstance(result, Path)
    assert "heart.csv" in str(result)
