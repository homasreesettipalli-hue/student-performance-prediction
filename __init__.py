"""
Student Performance Prediction and Early Warning System Package.
"""

from src.feature_engineering import calculate_lbi, compute_quiz_performance
from src.parent_alert import generate_parent_alert_content, is_demo_mode, send_parent_alert_email
from src.prediction import StudentPredictor
from src.preprocessing import StudentDataPreprocessor, prepare_training_data
from src.recommendations import classify_risk, generate_recommendations, identify_weak_areas

__all__ = [
    "calculate_lbi",
    "compute_quiz_performance",
    "StudentDataPreprocessor",
    "prepare_training_data",
    "classify_risk",
    "generate_recommendations",
    "identify_weak_areas",
    "StudentPredictor",
    "generate_parent_alert_content",
    "send_parent_alert_email",
    "is_demo_mode",
]

