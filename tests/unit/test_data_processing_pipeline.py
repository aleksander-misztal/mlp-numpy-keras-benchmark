# import pytest
# import pandas as pd
# import numpy as np
# from sklearn.pipeline import Pipeline
# from src.pipelines.data_processing_pipeline import build_pipeline
# from src.pipelines.data_processing_pipeline import DataCleaningTransformer


# @pytest.fixture
# def sample_df():
#     """
#     Provides a representative sample DataFrame for testing the DataCleaningTransformer.
#     Covers categorical, numerical, and missing value scenarios.
#     """
#     return pd.DataFrame({
#         'Age': [40, 60, 50],
#         'Sex': ['M', 'F', 'M'],
#         'FastingBS': [0, 1, 0],
#         'ExerciseAngina': ['Y', 'N', 'Y'],
#         'ChestPainType': ['ATA', 'NAP', 'ASY'],
#         'RestingECG': ['Normal', 'ST', 'LVH'],
#         'ST_Slope': ['Up', 'Flat', 'Down'],
#         'RestingBP': [120, 140, 130],
#         'Cholesterol': [200, np.nan, 180],
#         'Oldpeak': [1.2, np.nan, 0.6],
#         'MaxHR': [150, 130, 160]
#     })


# def test_fit_transform_runs(sample_df):
#     """
#     Ensures fit and transform run without error and produce expected output columns.
#     """
#     transformer = DataCleaningTransformer()
#     transformer.fit(sample_df)
#     transformed = transformer.transform(sample_df)

#     assert isinstance(transformed, pd.DataFrame)
#     assert "Sex_FastingBS_Freq" in transformed.columns  # interaction feature
#     assert "CholesterolPerAge" in transformed.columns   # derived feature
#     assert "HRRatio" in transformed.columns             # derived feature
#     assert not transformed.isnull().any().any()         # all NaNs should be handled


# def test_imputation_flags(sample_df):
#     """
#     Validates that missing value flags are added and correctly identify NaNs.
#     """
#     transformer = DataCleaningTransformer()
#     transformer.fit(sample_df)
#     transformed = transformer.transform(sample_df)

#     assert "Cholesterol_was_empty" in transformed.columns
#     assert "Oldpeak_was_empty" in transformed.columns
#     assert set(transformed["Cholesterol_was_empty"]) == {0, 1}


# def test_outlier_removal_does_not_crash(sample_df):
#     """
#     Tests that outlier logic handles extreme values gracefully.
#     """
#     df = sample_df.copy()
#     df.loc[0, 'RestingBP'] = 1000  # simulate outlier
#     transformer = DataCleaningTransformer()
#     transformer.fit(df)
#     transformed = transformer.transform(df)

#     assert transformed.shape[0] == df.shape[0]  # transformer shouldn't drop rows
#     assert "RestingBP" in transformed.columns


# def test_pipeline_works(sample_df):
#     """
#     Validates that pipeline constructed with build_pipeline() works end-to-end.
#     """
#     pipeline = build_pipeline()
#     transformed = pipeline.fit_transform(sample_df, y=None)

#     assert isinstance(transformed, pd.DataFrame)
#     assert "Sex_FastingBS_Freq" in transformed.columns
