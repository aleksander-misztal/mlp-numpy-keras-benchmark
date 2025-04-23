from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
import logging

logger = logging.getLogger(__name__)


def build_keras_mlp(input_dim: int, lr: float = 0.001, dropout: float = 0.2) -> Sequential:
    """
    Builds and compiles a simple MLP model using Keras Sequential API.

    Args:
        input_dim (int): Number of input features.
        lr (float): Learning rate for the Adam optimizer.
        dropout (float): Dropout rate between dense layers.

    Returns:
        keras.models.Sequential: Compiled Keras MLP model.
    """
    try:
        model = Sequential([
            Dense(32, activation='relu', input_shape=(input_dim,)),
            Dropout(dropout),
            Dense(16, activation='relu'),
            Dropout(dropout),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=BinaryCrossentropy(),
            metrics=[BinaryAccuracy()]
        )

        logger.info(f"Keras MLP built successfully (input_dim={input_dim}, lr={lr}, dropout={dropout})")
        return model

    except Exception as e:
        logger.exception("Failed to build Keras MLP model.")
        raise RuntimeError("Keras MLP model building failed.") from e
