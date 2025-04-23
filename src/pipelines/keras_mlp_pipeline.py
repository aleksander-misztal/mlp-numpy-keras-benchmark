from src.pipelines.base_mlp_pipeline import BaseMLPPipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
import numpy as np
import pandas as pd


class KerasMLPPipeline(BaseMLPPipeline):
    """
    MLP pipeline using Keras as backend.
    Supports training, prediction, evaluation and serialization.
    """

    def __init__(self, lr=0.001, batch=32, epochs=100, dropout=0.2):
        """
        Initialize training configuration.
        """
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.dropout = dropout

        self.model = None
        self.metrics = {}
        self.y_test = None
        self.y_proba = None

    def build_model(self, input_dim):
        """
        Build a simple feedforward neural network.

        Args:
            input_dim (int): Number of input features.

        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        model = Sequential([
            Dense(32, activation='relu', input_shape=(input_dim,)),
            Dropout(self.dropout),
            Dense(16, activation='relu'),
            Dropout(self.dropout),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss=BinaryCrossentropy(),
            metrics=[BinaryAccuracy()]
        )
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model on training data.

        Args:
            X_train, y_train: Training dataset.
            X_val, y_val (optional): Validation set.
        """
        self.model = self.build_model(input_dim=X_train.shape[1])
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=self.epochs,
            batch_size=self.batch,
            verbose=1
        )

    def predict(self, X):
        """
        Predict binary labels for the input features.
        """
        preds = self.model.predict(X)
        return (preds >= 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """
        Predict class probabilities (sigmoid outputs).
        """
        return self.model.predict(X).flatten()

    def evaluate(self, X, y):
        """
        Evaluate the model using Keras' built-in evaluation.

        Returns:
            Tuple of loss and accuracy.
        """
        return self.model.evaluate(X, y, verbose=0)

    def save_model(self, path):
        """
        Save model in TensorFlow SavedModel format.
        """
        self.model.save(path)

    def load_model(self, path_or_uri):
        """
        Load a trained model from file or URI.
        """
        from tensorflow.keras.models import load_model
        self.model = load_model(path_or_uri)

    def run(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Execute the full training and evaluation pipeline.

        Args:
            X_train, y_train: Training set.
            X_val, y_val: Validation set.
            X_test, y_test: Test set.
        """
        self.train(X_train, y_train, X_val, y_val)

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        self.y_test = y_test
        self.y_proba = y_proba

        self.metrics = self._compute_metrics(y_test, y_pred, y_proba)
