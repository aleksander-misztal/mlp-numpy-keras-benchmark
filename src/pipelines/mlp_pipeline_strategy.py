from src.pipelines.keras_mlp_pipeline import KerasMLPPipeline
from src.pipelines.numpy_mlp_pipeline import NumpyMLPPipeline

import logging
from typing import Type

# --- Basic logging setup for strategy selection ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class MLPPipelineStrategy:
    """
    Strategy pattern for selecting MLP pipeline implementations.
    
    Maps backend identifiers (e.g., 'keras', 'numpy') to corresponding 
    pipeline class implementations (e.g., `KerasMLPPipeline`).
    """

    def __init__(self):
        """
        Initializes the registry that maps model names to pipeline classes.
        """
        # A registry of available backend strategies (model names to pipeline classes)
        self.registry: dict[str, Type] = {
            "keras": KerasMLPPipeline,
            "numpy": NumpyMLPPipeline,
            # "torch": TorchMLPPipeline,  # Placeholder for future backend
        }

    def get_pipeline_class(self, model_name: str) -> Type:
        """
        Retrieve the pipeline class corresponding to the specified model backend.

        Args:
            model_name (str): Identifier of the backend model (e.g., 'keras', 'numpy').

        Returns:
            Type: The class implementing BaseMLPPipeline for the specified backend.

        Raises:
            ValueError: If the specified model name is not registered in the strategy registry.
        """
        # Normalize the model name to lowercase for case-insensitive comparison
        model_name = model_name.lower()
        
        if model_name not in self.registry:
            # Log the error and raise an exception if model name is not found
            available = list(self.registry.keys())
            logger.error(f"Strategy '{model_name}' is not recognized. Available: {available}")
            raise ValueError(f"Unknown strategy: '{model_name}'. Available: {available}")

        logger.info(f"Selected pipeline strategy: '{model_name}'")
        return self.registry[model_name]
