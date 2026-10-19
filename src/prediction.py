"""
Prediction Pipeline Module.
Connects preprocessor, trained regression model, feature engineering,
risk assessment, and recommendation generation into a single callable workflow.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np

from src.feature_engineering import calculate_lbi, compute_quiz_performance
from src.preprocessing import StudentDataPreprocessor
from src.recommendations import classify_risk, generate_recommendations, identify_weak_areas


class StudentPredictor:
    """
    Handles end-to-end inference from raw student inputs to score, risk, and recommendations.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        preprocessor_path: Union[str, Path],
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {self.model_path}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")

        self.model = joblib.load(self.model_path)
        self.preprocessor: StudentDataPreprocessor = joblib.load(self.preprocessor_path)

    def predict(self, raw_inputs: Dict[str, float]) -> Dict[str, Any]:
        """
        Executes complete prediction workflow:
        1. Input validation & default fills
        2. LBI calculation
        3. Feature scaling through fitted preprocessor
        4. Model prediction (clipped to [0, 100])
        5. Risk tier classification
        6. Weak areas identification
        7. Dynamic recommendation generation
        """
        # Validate and extract inputs
        gpa = float(raw_inputs.get("previous_gpa", 7.0))
        internal = float(raw_inputs.get("internal_marks", 65.0))
        prev_sub = float(raw_inputs.get("previous_subject_marks", 65.0))
        quiz_marks = float(raw_inputs.get("quiz_marks", 65.0))
        assign_marks = float(raw_inputs.get("assignment_marks", 70.0))

        attendance = float(raw_inputs.get("attendance", 75.0))
        study_hours = float(raw_inputs.get("study_hours", 3.0))
        assign_comp = float(raw_inputs.get("assignment_completion", 80.0))
        quiz_part = float(raw_inputs.get("quiz_participation", 80.0))
        learning_act = float(raw_inputs.get("learning_activity", 60.0))

        # Compute Quiz Performance & LBI
        quiz_perf = compute_quiz_performance(quiz_marks, quiz_part)
        lbi_value = calculate_lbi(
            quiz_performance=quiz_perf,
            assignment_completion=assign_comp,
            attendance=attendance,
            study_hours=study_hours,
            learning_activity=learning_act,
        )

        # Prepare feature vector dictionary exactly matching FEATURE_COLUMNS
        feature_dict = {
            "previous_gpa": gpa,
            "internal_marks": internal,
            "previous_subject_marks": prev_sub,
            "quiz_marks": quiz_marks,
            "assignment_marks": assign_marks,
            "attendance": attendance,
            "study_hours": study_hours,
            "assignment_completion": assign_comp,
            "quiz_participation": quiz_part,
            "learning_activity": learning_act,
            "lbi": lbi_value,
        }

        # Scale features
        X_scaled = self.preprocessor.transform(feature_dict)

        # Model Inference
        raw_pred = float(self.model.predict(X_scaled)[0])
        predicted_score = round(float(np.clip(raw_pred, 0.0, 100.0)), 2)

        # Risk Classification
        risk_level, risk_color, risk_icon = classify_risk(predicted_score)

        # Weak Area Detection
        weak_areas = identify_weak_areas(feature_dict)

        # Personalized Recommendations
        recommendations = generate_recommendations(
            feature_dict, predicted_score, weak_areas
        )

        return {
            "predicted_score": predicted_score,
            "lbi": lbi_value,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "risk_icon": risk_icon,
            "weak_areas": weak_areas,
            "recommendations": recommendations,
            "input_features": feature_dict,
        }
