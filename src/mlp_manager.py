import os
import logging
from src.utils.ml_utils import MLUtils
from src.pipelines.data_processing_pipeline import build_pipeline
from src.utils.data_utils import DataUtils

logger = logging.getLogger(__name__)

class MLPManager:
    """
    Manages the full MLP training and inference workflow, including:
    - Data loading and preprocessing
    - Model training or reuse
    - Performance evaluation and reporting
    """

    def __init__(
        self,
        model: str = "keras",
        lr: float = 0.01,
        batch: int = 12,
        epochs: int = 200,
        dropout: float = 0.1,
        output: str = "src/models",
        experiment: str = "HeartModelTraining"
    ):
        """
        Initialize training configuration and prepare ML utility class.
        """
        self.config = {
            "model": model,
            "lr": lr,
            "batch": batch,
            "epochs": epochs,
            "dropout": dropout,
            "output": output,
            "experiment": experiment
        }

        self.ml = MLUtils(self.config)
        self.pipeline = None
        self.model_file = None

    def run(self):
        """
        Execute the end-to-end training pipeline.
        Loads data, checks for existing model, trains if needed, and stores results.
        """
        logger.info("Running MLPManager...")

        self.ml.create_pipeline()
        X_train, y_train, X_val, y_val, X_test, y_test = self._load_data()

        if self.ml.model_exists():
            # Load and evaluate existing model
            self.pipeline, self.model_file = self.ml.load_model_and_predict(X_test, y_test)
        else:
            # Train new model and evaluate
            self.pipeline, self.model_file = self.ml.train_and_return(
                X_train, y_train, X_val, y_val, X_test, y_test
            )

    def _load_data(self):
        """
        Downloads dataset (if needed) and applies preprocessing pipeline.
        """
        raw_data_path = DataUtils.source_raw_data_from_kaggle()
        return DataUtils.load_and_process_data(path=raw_data_path, pipeline_builder=build_pipeline)

    def get_results(self) -> dict:
        """
        Returns evaluation metrics and visualizations from the trained pipeline.

        Raises:
            RuntimeError: if the pipeline hasn't been run.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call run() first.")

        try:
            return {
                "accuracy": self.pipeline.metrics["accuracy"],
                "roc_auc": self.pipeline.metrics["roc_auc"],
                "confusion_matrix": self.pipeline.metrics["confusion_matrix"],
                "classification_report": self.pipeline.metrics["classification_report"],
                "confusion_matrix_plot": self.pipeline.plot_confusion_matrix(),
                "roc_curve_plot": self.pipeline.plot_roc_curve(
                    self.pipeline.y_test, self.pipeline.y_proba
                )
            }
        except Exception as e:
            logger.error(f"Error while retrieving results: {e}")
            raise RuntimeError("Failed to collect results from pipeline.") from e
