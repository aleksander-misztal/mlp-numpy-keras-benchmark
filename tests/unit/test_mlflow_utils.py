# tests/unit/test_mlflow_utils.py

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.utils.mlflow_utils import MLflowUtils

@patch("src.utils.mlflow_utils.MlflowClient")
def test_get_model_ranking_returns_dataframe(mock_mlflow_client_class):
    # Arrange
    mock_client = MagicMock()
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "123"

    mock_run = MagicMock()
    mock_run.info.run_id = "run_1"
    mock_run.data.params = {"model": "keras", "lr": "0.001", "epochs": "50", "dropout": "0.1"}
    mock_run.data.metrics = {"accuracy": 0.9, "roc_auc": 0.88, "training_time_sec": 15.2}

    mock_client.get_experiment_by_name.return_value = mock_experiment
    mock_client.search_runs.return_value = [mock_run]
    mock_mlflow_client_class.return_value = mock_client

    # Act
    df = MLflowUtils.get_model_ranking()

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.loc[0, "Model"] == "keras"
    assert df.loc[0, "Accuracy"] == 0.9

@patch("src.utils.mlflow_utils.MlflowClient")
def test_get_model_ranking_returns_empty_when_no_experiment(mock_mlflow_client_class):
    # Arrange
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = None
    mock_mlflow_client_class.return_value = mock_client

    # Act
    df = MLflowUtils.get_model_ranking()

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert df.empty
