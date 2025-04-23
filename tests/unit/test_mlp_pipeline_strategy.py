# tests/unit/test_mlp_pipeline_strategy.py

import pytest
from src.pipelines.mlp_pipeline_strategy import MLPPipelineStrategy
from src.pipelines.keras_mlp_pipeline import KerasMLPPipeline
from src.pipelines.numpy_mlp_pipeline import NumpyMLPPipeline

def test_get_pipeline_class_returns_correct_class():
    strategy = MLPPipelineStrategy()

    keras_cls = strategy.get_pipeline_class("keras")
    numpy_cls = strategy.get_pipeline_class("numpy")

    assert keras_cls is KerasMLPPipeline
    assert numpy_cls is NumpyMLPPipeline

def test_get_pipeline_class_raises_on_unknown_strategy():
    strategy = MLPPipelineStrategy()

    with pytest.raises(ValueError, match="Unknown strategy: 'torch'"):
        strategy.get_pipeline_class("torch")
