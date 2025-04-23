from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd

class MLflowUtils:
    """
    Utility class for interacting with MLflow experiments, runs, and metrics.
    """

    @staticmethod
    def get_model_ranking(experiment_name="HeartModelTraining", metric="accuracy", top_n=10) -> pd.DataFrame:
        """
        Fetch top N runs from MLflow based on a given metric.

        Args:
            experiment_name (str): Name of the MLflow experiment.
            metric (str): Metric to rank by (e.g., 'accuracy').
            top_n (int): Number of top runs to return.

        Returns:
            pd.DataFrame: Ranked runs with key metadata and metrics.
        """
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return pd.DataFrame()

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=top_n
        )

        # Extract relevant details for display
        data = []
        for run in runs:
            data.append({
                "Run ID": run.info.run_id,
                "Model": run.data.params.get("model"),
                "LR": run.data.params.get("lr"),
                "Epochs": run.data.params.get("epochs"),
                "Dropout": run.data.params.get("dropout"),
                "Accuracy": run.data.metrics.get("accuracy"),
                "ROC AUC": run.data.metrics.get("roc_auc"),
                "Training Time (s)": run.data.metrics.get("training_time_sec")
            })

        return pd.DataFrame(data)

    @staticmethod
    def log_run(model_name: str, params: dict, metrics: dict, model_path: str, experiment: str) -> None:
        """
        Log a training run to MLflow with parameters, metrics, and artifacts.

        Args:
            model_name (str): Name of the model (e.g., 'keras', 'numpy').
            params (dict): Training parameters.
            metrics (dict): Evaluation metrics.
            model_path (str): Path to the trained model file.
            experiment (str): Target MLflow experiment name.
        """
        mlflow.set_experiment(experiment)
        with mlflow.start_run():
            mlflow.log_params({"model": model_name, **params})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(model_path, artifact_path="model_file")
            mlflow.set_tag("source", "trained")
