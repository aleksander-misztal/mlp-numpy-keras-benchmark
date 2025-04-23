import os
import time
import logging
from src.pipelines.mlp_pipeline_strategy import MLPPipelineStrategy
from src.utils.mlflow_utils import MLflowUtils

logger = logging.getLogger(__name__)

class MLUtils:
    """
    Utility class to manage the lifecycle of an MLP training pipeline:
    - Selects implementation (Keras or NumPy)
    - Handles training, saving, and loading models
    - Logs results to MLflow
    """

    def __init__(self, config: dict):
        """
        Initialize the utility with training configuration.

        Args:
            config (dict): Model training configuration.
        """
        self.model = config["model"]
        self.lr = config["lr"]
        self.batch = config["batch"]
        self.epochs = config["epochs"]
        self.dropout = config["dropout"]
        self.output = config["output"]
        self.experiment = config["experiment"]

        self.pipeline = None
        self.model_file = os.path.join(
            self.output,
            f"{self.model}_lr{self.lr}_ep{self.epochs}_do{self.dropout}.h5"
        )

        os.makedirs(self.output, exist_ok=True)
        self.strategy = MLPPipelineStrategy()

    def create_pipeline(self):
        """
        Instantiate the selected pipeline class (NumPy or Keras).
        """
        pipeline_class = self.strategy.get_pipeline_class(self.model)
        self.pipeline = pipeline_class(
            lr=self.lr,
            batch=self.batch,
            epochs=self.epochs,
            dropout=self.dropout
        )
        logger.info(f"Pipeline created: {pipeline_class.__name__}")

    def model_exists(self) -> bool:
        """
        Check if the model file already exists (to avoid retraining).
        """
        return os.path.exists(self.model_file)

    def load_model_and_predict(self, X_test, y_test):
        """
        Load a saved model and perform inference on test data.

        Args:
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Tuple of pipeline and model file path.

        Raises:
            RuntimeError: if loading or prediction fails.
        """
        try:
            logger.info(f"Loading model from: {self.model_file}")
            self.pipeline.load_model(self.model_file)
            self.pipeline.y_test = y_test
            self.pipeline.y_proba = self.pipeline.predict_proba(X_test)
            y_pred = self.pipeline.predict(X_test)
            self.pipeline.metrics = self.pipeline._compute_metrics(
                y_test, y_pred, self.pipeline.y_proba
            )
            return self.pipeline, self.model_file
        except Exception as e:
            logger.error(f"Error during model loading or inference: {e}")
            raise RuntimeError("Failed to load model or predict.") from e

    def train_and_return(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Train a new model, save it, and log the run to MLflow.

        Args:
            X_train, y_train, X_val, y_val, X_test, y_test: Dataset splits.

        Returns:
            Tuple of pipeline and model file path.
        """
        logger.info("Training new model...")
        start_time = time.time()

        self.pipeline.run(X_train, y_train, X_val, y_val, X_test, y_test)
        duration = time.time() - start_time
        logger.info(f"Training completed in {duration:.2f} seconds.")

        self.pipeline.save_model(self.model_file)

        # Log run to MLflow
        MLflowUtils.log_run(
            model_name=self.model,
            params={
                "lr": self.lr,
                "batch": self.batch,
                "epochs": self.epochs,
                "dropout": self.dropout
            },
            metrics={
                "accuracy": self.pipeline.metrics["accuracy"],
                "roc_auc": self.pipeline.metrics["roc_auc"],
                "training_time_sec": duration
            },
            model_path=self.model_file,
            experiment=self.experiment
        )

        return self.pipeline, self.model_file
