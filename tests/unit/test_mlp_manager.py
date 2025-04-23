import pytest
from unittest.mock import MagicMock, patch
from src.mlp_manager import MLPManager
from src.utils.ml_utils import MLUtils

@pytest.fixture
def sample_config():
    return {
        "model": "keras",
        "lr": 0.01,
        "batch": 16,
        "epochs": 50,
        "dropout": 0.1,
        "output": "src/models",
        "experiment": "TestExperiment"
    }

@patch("src.utils.ml_utils.MLUtils.load_model_and_predict")
@patch("src.utils.ml_utils.MLUtils.model_exists", return_value=True)
@patch("src.utils.ml_utils.MLUtils.create_pipeline")
@patch("src.utils.data_utils.DataUtils.load_and_process_data", return_value=("Xtr", "ytr", "Xval", "yval", "Xt", "yt"))
@patch("src.utils.data_utils.DataUtils.source_raw_data_from_kaggle", return_value="dummy.csv")
def test_manager_run_with_cached_model(mock_kaggle, mock_loader, mock_create, mock_exists, mock_load_predict, sample_config):
    manager = MLPManager(**sample_config)
    manager.ml = MLUtils(sample_config)
    manager.ml.create_pipeline = MagicMock()
    manager.ml.model_exists = MagicMock(return_value=True)
    manager.ml.load_model_and_predict = MagicMock(return_value=(MagicMock(), "model.h5"))

    manager.run()

    manager.ml.load_model_and_predict.assert_called_once()
    assert manager.pipeline is not None
    assert manager.model_file == "model.h5"
