import numpy as np
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class NumpyMLP:
    """
    A NumPy-based implementation of a Multi-Layer Perceptron (MLP).
    Supports two hidden layers, dropout, mini-batch training, and binary classification.
    """

    @staticmethod
    def initialize(input_size: int, hidden1: int = 32, hidden2: int = 16, seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Initializes network parameters using He initialization.

        Args:
            input_size (int): Number of input features.
            hidden1 (int): Number of units in the first hidden layer.
            hidden2 (int): Number of units in the second hidden layer.
            seed (int): Random seed for reproducibility.

        Returns:
            dict: Dictionary containing weights and biases for each layer.
        """
        np.random.seed(seed)
        return {
            'W1': np.random.randn(input_size, hidden1) * np.sqrt(2. / input_size),
            'b1': np.zeros(hidden1),
            'W2': np.random.randn(hidden1, hidden2) * np.sqrt(2. / hidden1),
            'b2': np.zeros(hidden2),
            'W3': np.random.randn(hidden2, 1) * np.sqrt(2. / hidden2),
            'b3': np.zeros(1),
        }

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function.
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x: np.ndarray) -> np.ndarray:
        """
        Derivative of the ReLU activation function.
        """
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid activation function.
        """
        s = NumpyMLP.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary crossentropy loss between true and predicted values.
        
        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.
        
        Returns:
            float: Binary crossentropy loss.
        """
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def forward(X: np.ndarray, params: Dict[str, np.ndarray],
                training: bool = True, dropout_rate: float = 0.2) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Perform a forward pass through the MLP.

        Args:
            X (np.ndarray): Input data.
            params (dict): Network parameters (weights and biases).
            training (bool): Whether the model is being trained (for dropout).
            dropout_rate (float): Dropout rate for regularization.

        Returns:
            Tuple: 
                - Final output after the forward pass.
                - Cache of intermediate values for backpropagation.
        """
        Z1 = X @ params['W1'] + params['b1']
        A1 = NumpyMLP.relu(Z1)

        D1 = np.ones_like(A1)
        if training:
            D1 = (np.random.rand(*A1.shape) > dropout_rate).astype(float)
            A1 *= D1
            A1 /= (1.0 - dropout_rate)

        Z2 = A1 @ params['W2'] + params['b2']
        A2 = NumpyMLP.relu(Z2)

        D2 = np.ones_like(A2)
        if training:
            D2 = (np.random.rand(*A2.shape) > dropout_rate).astype(float)
            A2 *= D2
            A2 /= (1.0 - dropout_rate)

        Z3 = A2 @ params['W3'] + params['b3']
        A3 = NumpyMLP.sigmoid(Z3)

        cache = {
            'X': X, 'Z1': Z1, 'A1': A1, 'D1': D1,
            'Z2': Z2, 'A2': A2, 'D2': D2,
            'Z3': Z3, 'A3': A3
        }

        return A3, cache

    @staticmethod
    def backward(y_true: np.ndarray, cache: Dict[str, np.ndarray],
                 params: Dict[str, np.ndarray], learning_rate: float = 0.001, dropout_rate: float = 0.2) -> None:
        """
        Backpropagate the error and update the network's parameters.

        Args:
            y_true (np.ndarray): True labels.
            cache (dict): Cache of intermediate values from the forward pass.
            params (dict): Network parameters (weights and biases).
            learning_rate (float): Learning rate for parameter updates.
            dropout_rate (float): Dropout rate used during training.
        """
        m = y_true.shape[0]
        A3 = cache['A3']
        A2 = cache['A2']
        A1 = cache['A1']
        X = cache['X']
        D1 = cache['D1']
        D2 = cache['D2']

        dZ3 = A3 - y_true.reshape(-1, 1)
        dW3 = A2.T @ dZ3 / m
        db3 = np.mean(dZ3, axis=0)

        dA2 = dZ3 @ params['W3'].T * NumpyMLP.relu_deriv(cache['Z2']) * D2 / (1.0 - dropout_rate)
        dW2 = A1.T @ dA2 / m
        db2 = np.mean(dA2, axis=0)

        dA1 = dA2 @ params['W2'].T * NumpyMLP.relu_deriv(cache['Z1']) * D1 / (1.0 - dropout_rate)
        dW1 = X.T @ dA1 / m
        db1 = np.mean(dA1, axis=0)

        # Update parameters
        params['W3'] -= learning_rate * dW3
        params['b3'] -= learning_rate * db3
        params['W2'] -= learning_rate * dW2
        params['b2'] -= learning_rate * db2
        params['W1'] -= learning_rate * dW1
        params['b1'] -= learning_rate * db1

    @staticmethod
    def predict(X: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Make binary predictions (0 or 1) using the trained model.

        Args:
            X (np.ndarray): Input features for prediction.
            params (dict): Trained model parameters.

        Returns:
            np.ndarray: Binary predictions (0 or 1).
        """
        y_prob, _ = NumpyMLP.forward(X, params, training=False)
        return (y_prob >= 0.5).astype(int).flatten()

    @staticmethod
    def predict_proba(X: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities for the positive class.

        Args:
            X (np.ndarray): Input features for prediction.
            params (dict): Trained model parameters.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        y_prob, _ = NumpyMLP.forward(X, params, training=False)
        return y_prob.flatten()

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 150, batch_size: int = 10, learning_rate: float = 0.001,
            dropout_rate: float = 0.2, verbose: int = 1) -> Dict[str, np.ndarray]:
        """
        Train the MLP on the given data using mini-batch gradient descent.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Labels.
            X_val (np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation labels.
            epochs (int): Number of training epochs.
            batch_size (int): Size of mini-batches.
            learning_rate (float): Learning rate for training.
            dropout_rate (float): Dropout rate for regularization.
            verbose (int): Verbosity level (e.g., 1 for logging every 10 epochs).

        Returns:
            dict: Trained model parameters.
        """
        params = NumpyMLP.initialize(input_size=X.shape[1])
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                y_pred, cache = NumpyMLP.forward(X_batch, params, training=True, dropout_rate=dropout_rate)
                NumpyMLP.backward(y_batch, cache, params, learning_rate, dropout_rate)

            if verbose and epoch % 10 == 0:
                train_pred = NumpyMLP.predict_proba(X, params)
                train_loss = NumpyMLP.binary_crossentropy(y, train_pred)
                train_acc = np.mean((train_pred >= 0.5).astype(int) == y)

                if X_val is not None and y_val is not None:
                    val_pred = NumpyMLP.predict_proba(X_val, params)
                    val_loss = NumpyMLP.binary_crossentropy(y_val, val_pred)
                    val_acc = np.mean((val_pred >= 0.5).astype(int) == y_val)
                    logger.info(f"[Epoch {epoch}] loss={train_loss:.4f} acc={train_acc:.4f} "
                                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
                else:
                    logger.info(f"[Epoch {epoch}] loss={train_loss:.4f} acc={train_acc:.4f}")

        return params
