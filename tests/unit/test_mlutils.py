import pytest
from unittest.mock import MagicMock, patch
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

def test_mlutils_model_exists_true(tmp_path, sample_config):
    sample_config["output"] = str(tmp_path)
    model_file = tmp_path / "keras_lr0.01_ep50_do0.1.h5"
    model_file.write_text("dummy model content")
    ml = MLUtils(sample_config)
    assert ml.model_exists() is True

def test_mlutils_model_exists_false(tmp_path, sample_config):
    sample_config["output"] = str(tmp_path)
    ml = MLUtils(sample_config)
    assert ml.model_exists() is False

@patch("src.utils.ml_utils.MLflowUtils.log_run")
def test_mlutils_train_and_return(mock_log_run, sample_config):
    ml = MLUtils(sample_config)
    ml.pipeline = MagicMock()
    ml.pipeline.metrics = {
        "accuracy": 0.95,
        "roc_auc": 0.9
    }
    ml.pipeline.run = MagicMock()
    ml.pipeline.save_model = MagicMock()

    pipeline, path = ml.train_and_return(
        X_train="Xtr", y_train="ytr",
        X_val="Xval", y_val="yval",
        X_test="Xt", y_test="yt"
    )

    assert pipeline is ml.pipeline
    assert path.endswith(".h5")
    mock_log_run.assert_called_once()
