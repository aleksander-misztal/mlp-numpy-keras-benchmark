from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Any

logger = logging.getLogger(__name__)


class BaseMLPPipeline(ABC):
    """
    Abstract base class for MLP model pipelines. Defines core interface and
    provides shared metric evaluation and plotting functionality.
    """

    @abstractmethod
    def train(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None) -> None:
        """
        Trains the model.
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Predicts class labels from input features.
        """
        pass

    @abstractmethod
    def evaluate(self, X_test: Any, y_test: Any) -> Any:
        """
        Evaluates model performance.
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Persists model to the given path.
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Loads model from the given path.
        """
        pass

    def _compute_metrics(self, y_test, y_pred, y_proba) -> dict:
        """
        Computes and returns evaluation metrics.

        Returns:
            dict: Includes accuracy, ROC AUC, confusion matrix, classification report.
        """
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": pd.DataFrame(
                classification_report(y_test, y_pred, output_dict=True)
            ).transpose()
        }

    def plot_confusion_matrix(self, labels=("Negative", "Positive"), title="Confusion Matrix"):
        """
        Plots confusion matrix stored in self.metrics.

        Args:
            labels (tuple): Class labels for axis ticks.
            title (str): Title for the plot.

        Returns:
            matplotlib.figure.Figure
        """
        try:
            cm = getattr(self, "metrics", {}).get("confusion_matrix")
            if cm is None:
                raise ValueError("Confusion matrix not available. Run the model first.")

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(title)
            return fig

        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")
            raise

    def plot_roc_curve(self, y_true, y_scores, title="ROC Curve"):
        """
        Plots ROC curve for model predictions.

        Args:
            y_true (array-like): Ground truth labels.
            y_scores (array-like): Predicted probabilities.
            title (str): Title for the plot.

        Returns:
            matplotlib.figure.Figure
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label="ROC Curve")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
            ax.set_title(title)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            return fig

        except Exception as e:
            logger.error(f"Failed to plot ROC curve: {e}")
            raise
