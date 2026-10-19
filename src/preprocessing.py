"""
Preprocessing Module for Student Performance Prediction.
Handles data cleaning, missing value imputation, duplicate removal, feature scaling,
and train-test splits in a modular, reproducible manner.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLUMNS = [
    "previous_gpa",
    "internal_marks",
    "previous_subject_marks",
    "quiz_marks",
    "assignment_marks",
    "attendance",
    "study_hours",
    "assignment_completion",
    "quiz_participation",
    "learning_activity",
    "lbi",
]


class StudentDataPreprocessor:
    """
    Encapsulates preprocessing logic:
    1. Column selection and ordering
    2. Imputation of missing values
    3. Numerical feature standardization
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates and basic anomalies.
        """
        df_clean = df.copy()
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        dropped_duplicates = initial_count - len(df_clean)
        if dropped_duplicates > 0:
            print(f"Removed {dropped_duplicates} duplicate records.")

        return df_clean

    def fit(self, X: pd.DataFrame) -> "StudentDataPreprocessor":
        """
        Fit imputer and scaler on training feature matrix.
        """
        X_subset = X[self.feature_columns].copy()
        imputed_data = self.imputer.fit_transform(X_subset)
        self.scaler.fit(imputed_data)
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, Dict[str, float]]) -> np.ndarray:
        """
        Transform feature matrix or single input dictionary using fitted imputer and scaler.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")

        if isinstance(X, dict):
            # Convert single dictionary to 1-row DataFrame
            X = pd.DataFrame([X])

        # Ensure all required columns exist
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Input is missing required features: {missing_cols}")

        X_subset = X[self.feature_columns].copy()
        imputed_data = self.imputer.transform(X_subset)
        scaled_data = self.scaler.transform(imputed_data)
        return scaled_data

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        """
        return self.fit(X).transform(X)

    def save(self, file_path: str) -> None:
        """
        Serialize preprocessor to disk using joblib.
        """
        joblib.dump(self, file_path)
        print(f"Saved preprocessor to {file_path}")

    @staticmethod
    def load(file_path: str) -> "StudentDataPreprocessor":
        """
        Deserialize preprocessor from disk.
        """
        return joblib.load(file_path)


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = "final_score",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StudentDataPreprocessor]:
    """
    Full pipeline to clean, split, and preprocess training data.
    Ensures zero data leakage by fitting preprocessor solely on X_train.
    """
    preprocessor = StudentDataPreprocessor()
    df_cleaned = preprocessor.clean_raw_data(df)

    # Separate features and target
    X = df_cleaned[preprocessor.feature_columns]
    y = df_cleaned[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit solely on training data
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor
