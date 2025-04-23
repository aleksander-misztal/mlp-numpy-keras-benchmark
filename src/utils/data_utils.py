import kagglehub
import shutil
from pathlib import Path
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configure console logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DataUtils:
    """
    Utilities for dataset download, validation, and preprocessing.
    """

    @staticmethod
    def source_raw_data_from_kaggle() -> Path:
        """
        Downloads the heart disease dataset from Kaggle, if not already available locally.

        Returns:
            Path: Local path to the heart.csv file.
        """
        save_dir = Path('src/data')
        file_name = 'heart.csv'
        file_path = save_dir / file_name

        if file_path.exists():
            logger.info(f"Dataset already exists at {file_path}")
            return file_path

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = Path(kagglehub.dataset_download("fedesoriano/heart-failure-prediction"))

            # Copy first found CSV from the downloaded dataset
            for csv_file in dataset_path.glob('*.csv'):
                shutil.copy2(csv_file, file_path)
                break

            logger.info(f"Dataset successfully downloaded to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to download dataset from Kaggle: {e}")
            raise RuntimeError("Dataset download failed.") from e

    @staticmethod
    def load_and_process_data(path, pipeline_builder=None):
        """
        Loads a CSV dataset, applies optional preprocessing, and splits it into train/val/test.

        Args:
            path (str or Path): Path to the CSV dataset.
            pipeline_builder (callable, optional): A function that returns a preprocessing pipeline.

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

            logger.info(f"Loading dataset from {path}")
            df = pd.read_csv(path)

            if 'HeartDisease' not in df.columns:
                raise ValueError("Expected target column 'HeartDisease' not found.")

            X = df.drop(columns=['HeartDisease'])
            y = df['HeartDisease']

            # Split into train (60%), val (16%), and test (24%)
            logger.info("Splitting dataset into train/val/test...")
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)

            # Apply preprocessing pipeline if provided
            if pipeline_builder:
                logger.info("Applying preprocessing pipeline...")
                pipeline = pipeline_builder()
                X_train = pipeline.fit_transform(X_train, y_train)
                X_val = pipeline.transform(X_val)
                X_test = pipeline.transform(X_test)

            return X_train, y_train, X_val, y_val, X_test, y_test

        except Exception as e:
            logger.error(f"Error loading and processing dataset: {e}")
            raise
