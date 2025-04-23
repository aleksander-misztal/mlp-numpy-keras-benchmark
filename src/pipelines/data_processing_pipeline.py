import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from typing import Any

logger = logging.getLogger(__name__)


class DataCleaningTransformer(BaseEstimator, TransformerMixin):
    """
    Preprocessing transformer for structured tabular data.

    Applies:
    - Label encoding for binary categorical features
    - Frequency encoding for multi-categorical features
    - IQR-based outlier removal and median imputation
    - Feature interactions and derived ratios
    """

    BINARY_COLS = ['Sex', 'ExerciseAngina']
    FREQ_ENCODE_COLS = ['ChestPainType', 'RestingECG', 'ST_Slope']
    OUTLIER_COLS = ['RestingBP', 'Cholesterol', 'Oldpeak']

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.freq_enc_maps: dict[str, dict] = {}
        self.medians: dict[str, float] = {}
        self.sex_fbs_freq_map: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DataCleaningTransformer":
        """Fits encoders, mappings, and median values for later use in transform."""
        df = X.copy()
        try:
            self._fit_label_encoders(df)
            self._fit_frequency_encodings(df)
            self._fit_outlier_medians(df)
            self._fit_sex_fbs_interaction(df)
            logger.info("DataCleaningTransformer: Fit complete.")
        except Exception as e:
            logger.exception("Fitting failed.")
            raise RuntimeError("Failed during fit() in DataCleaningTransformer.") from e

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input data using fitted encoders and mappings."""
        df = X.copy()
        try:
            self._add_sex_fbs_interaction(df)
            self._apply_label_encodings(df)
            self._apply_frequency_encodings(df)
            self._handle_outliers(df)
            self._impute_medians(df)
            self._add_derived_features(df)
            logger.info("DataCleaningTransformer: Transform complete.")
        except Exception as e:
            logger.exception("Transformation failed.")
            raise RuntimeError("Failed during transform() in DataCleaningTransformer.") from e

        return df

    # --- Private helper methods ---

    def _fit_label_encoders(self, df: pd.DataFrame):
        for col in self.BINARY_COLS:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            le.fit(df[col])
            self.label_encoders[col] = le

    def _fit_frequency_encodings(self, df: pd.DataFrame):
        for col in self.FREQ_ENCODE_COLS:
            self.freq_enc_maps[col] = df[col].value_counts(normalize=True).to_dict()

    def _fit_outlier_medians(self, df: pd.DataFrame):
        for col in self.OUTLIER_COLS:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            filtered = df[col].where(df[col].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR))
            self.medians[col] = filtered.median()

    def _fit_sex_fbs_interaction(self, df: pd.DataFrame):
        combo = df['Sex'].astype(str) + '_' + df['FastingBS'].astype(str)
        self.sex_fbs_freq_map = combo.value_counts(normalize=True).to_dict()

    def _add_sex_fbs_interaction(self, df: pd.DataFrame):
        combo = df['Sex'].astype(str) + '_' + df['FastingBS'].astype(str)
        df['Sex_FastingBS_Freq'] = combo.map(self.sex_fbs_freq_map)

    def _apply_label_encodings(self, df: pd.DataFrame):
        for col, encoder in self.label_encoders.items():
            df[col] = encoder.transform(df[col].astype(str))

    def _apply_frequency_encodings(self, df: pd.DataFrame):
        for col, mapping in self.freq_enc_maps.items():
            df[col] = df[col].map(mapping)

    def _handle_outliers(self, df: pd.DataFrame):
        for col in self.OUTLIER_COLS:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].where(df[col].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR))

    def _impute_medians(self, df: pd.DataFrame):
        for col in self.OUTLIER_COLS:
            if col in ['Cholesterol', 'Oldpeak']:
                df[f'{col}_was_empty'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(self.medians[col])

    def _add_derived_features(self, df: pd.DataFrame):
        df['CholesterolPerAge'] = df['Cholesterol'] / df['Age']
        df['HRRatio'] = df['MaxHR'] / (220 - df['Age'])

from sklearn.pipeline import Pipeline

def build_pipeline() -> Pipeline:
    return Pipeline([
        ('clean_features', DataCleaningTransformer())
    ])
