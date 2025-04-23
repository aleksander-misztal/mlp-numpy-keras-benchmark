import numpy as np
import pickle
import pandas as pd
from src.models_architectures.numpy_mlp_arch import NumpyMLP
from src.pipelines.base_mlp_pipeline import BaseMLPPipeline


def to_numpy(X):
    """
    Convert input to a NumPy array if it's a sparse matrix or pandas object.
    """
    if hasattr(X, "toarray"):
        return X.toarray()
    if isinstance(X, pd.DataFrame):
        return X.values
    return X


class NumpyMLPPipeline(BaseMLPPipeline):
    """
    MLP pipeline implementation using a NumPy-based neural network backend.
    """

    def __init__(self, lr=0.001, batch=32, epochs=100, dropout=0.2):
        """
        Initialize training configuration.
        """
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.dropout = dropout

        self.params = None
        self.metrics = {}
        self.y_test = None
        self.y_proba = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the NumPy-based MLP model.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
            X_val (array-like, optional): Validation features.
            y_val (array-like, optional): Validation labels.
        """
        X_train = to_numpy(X_train).astype(np.float32)
        y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

        X_val = to_numpy(X_val).astype(np.float32) if X_val is not None else None
        y_val = y_val.to_numpy() if isinstance(y_val, pd.Series) else y_val

        self.params = NumpyMLP.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=self.epochs,
            batch_size=self.batch,
            learning_rate=self.lr,
            dropout_rate=self.dropout
        )

    def predict(self, X):
        """
        Predict class labels using the trained model.
        """
        X = to_numpy(X).astype(np.float32)
        return NumpyMLP.predict(X, self.params)

    def predict_proba(self, X):
        """
        Predict class probabilities using the trained model.
        """
        X = to_numpy(X).astype(np.float32)
        return NumpyMLP.predict_proba(X, self.params)

    def evaluate(self, X, y):
        """
        Evaluate model performance on provided data.

        Returns:
            dict: Contains accuracy and loss.
        """
        X = to_numpy(X).astype(np.float32)
        y = y.to_numpy() if isinstance(y, pd.Series) else y

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        acc = np.mean(y_pred == y)
        loss = NumpyMLP.binary_crossentropy(y, y_proba)
        return {"accuracy": acc, "loss": loss}

    def save_model(self, path):
        """
        Serialize model parameters to disk.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.params, f)

    def load_model(self, path):
        """
        Load model parameters from file.
        """
        with open(path, 'rb') as f:
            self.params = pickle.load(f)

    def run(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Full pipeline run: train → predict → evaluate.

        Args:
            X_train, y_train: Training set.
            X_val, y_val: Validation set.
            X_test, y_test: Test set.
        """
        self.train(X_train, y_train, X_val, y_val)

        y_test = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test
        X_test = to_numpy(X_test).astype(np.float32)

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        self.y_test = y_test
        self.y_proba = y_proba
        self.metrics = self._compute_metrics(y_test, y_pred, y_proba)
