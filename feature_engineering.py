"""
Feature Engineering Module for Student Performance Prediction.
Computes domain-specific metrics including the Learning Behavior Index (LBI).
"""

import numpy as np
import pandas as pd


def normalize_study_hours(study_hours: float, max_reference_hours: float = 10.0) -> float:
    """
    Normalize study hours to a 0-100 scale based on a reference daily maximum (default 10 hours).
    Capped at 100.0.
    """
    hours = max(0.0, float(study_hours))
    return min((hours / max_reference_hours) * 100.0, 100.0)


def normalize_learning_activity(activity: float) -> float:
    """
    Ensure learning/LMS activity level is within 0-100 scale.
    """
    return min(max(float(activity), 0.0), 100.0)


def compute_quiz_performance(quiz_marks: float, quiz_participation: float = 100.0) -> float:
    """
    Calculate composite quiz performance (0-100 scale) combining quiz marks and participation.
    Weighting: 70% quiz marks + 30% quiz participation.
    """
    marks = min(max(float(quiz_marks), 0.0), 100.0)
    part = min(max(float(quiz_participation), 0.0), 100.0)
    return 0.70 * marks + 0.30 * part


def calculate_lbi(
    quiz_performance: float,
    assignment_completion: float,
    attendance: float,
    study_hours: float,
    learning_activity: float,
) -> float:
    """
    Calculate the Learning Behavior Index (LBI) from 0 to 100 using the specified PRD formula:
    LBI = 0.30 * Quiz Performance
        + 0.25 * Assignment Completion
        + 0.20 * Attendance
        + 0.15 * Study Hours (Normalized)
        + 0.10 * Learning Activity (Normalized)
    """
    quiz_perf_clean = min(max(float(quiz_performance), 0.0), 100.0)
    assign_clean = min(max(float(assignment_completion), 0.0), 100.0)
    attend_clean = min(max(float(attendance), 0.0), 100.0)
    study_norm = normalize_study_hours(study_hours)
    activity_norm = normalize_learning_activity(learning_activity)

    lbi = (
        0.30 * quiz_perf_clean
        + 0.25 * assign_clean
        + 0.20 * attend_clean
        + 0.15 * study_norm
        + 0.10 * activity_norm
    )
    return round(float(np.clip(lbi, 0.0, 100.0)), 2)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich raw dataframe with engineered features including LBI and composite performance indicators.
    """
    df_out = df.copy()

    # Quiz performance composite
    if "quiz_participation" in df_out.columns and "quiz_marks" in df_out.columns:
        df_out["quiz_performance"] = (
            0.70 * df_out["quiz_marks"] + 0.30 * df_out["quiz_participation"]
        ).clip(0, 100)
    elif "quiz_marks" in df_out.columns:
        df_out["quiz_performance"] = df_out["quiz_marks"].clip(0, 100)

    # Compute LBI for each row
    df_out["study_hours_norm"] = (
        (df_out["study_hours"] / 10.0) * 100.0
    ).clip(0, 100)
    df_out["learning_activity_norm"] = df_out["learning_activity"].clip(0, 100)

    df_out["lbi"] = (
        0.30 * df_out["quiz_performance"]
        + 0.25 * df_out["assignment_completion"]
        + 0.20 * df_out["attendance"]
        + 0.15 * df_out["study_hours_norm"]
        + 0.10 * df_out["learning_activity_norm"]
    ).round(2).clip(0, 100)

    # Academic baseline composite (scaled to 100)
    if "previous_gpa" in df_out.columns:
        df_out["previous_gpa_pct"] = (df_out["previous_gpa"] * 10.0).clip(0, 100)

    return df_out
