"""
Dataset Generation Script for Student Performance Prediction.
Generates logically consistent, realistic student performance data based on
domain-specific academic and learning behavior relationships.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Ensure local src package is importable
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.feature_engineering import calculate_lbi, compute_quiz_performance


def generate_student_dataset(
    n_samples: int = 1500,
    random_state: int = 42,
    output_path: Path = None,
) -> pd.DataFrame:
    """
    Generate synthetic dataset of student academic metrics and learning behaviors.
    """
    np.random.seed(random_state)

    # 4 Student Profiles / Archetypes for realistic distributions:
    # 0: High Achievers (~25%)
    # 1: Consistent Average (~40%)
    # 2: Inconsistent / At-Risk (~20%)
    # 3: Severe Risk / Disengaged (~15%)
    profiles = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.25, 0.40, 0.20, 0.15])

    data = []

    for i, profile in enumerate(profiles):
        student_id = f"STU{10001 + i}"

        if profile == 0:  # High Achiever
            gpa = np.clip(np.random.normal(8.8, 0.5), 7.8, 10.0)
            attendance = np.clip(np.random.normal(90, 5), 78, 100)
            study_hours = np.clip(np.random.normal(5.0, 1.0), 3.0, 9.0)
            internal_marks = np.clip(np.random.normal(86, 6), 72, 100)
            prev_sub_marks = np.clip(np.random.normal(85, 7), 70, 100)
            quiz_marks = np.clip(np.random.normal(88, 6), 72, 100)
            assign_marks = np.clip(np.random.normal(90, 5), 75, 100)
            assign_comp = np.clip(np.random.normal(94, 4), 80, 100)
            quiz_part = np.clip(np.random.normal(95, 4), 80, 100)
            learning_act = np.clip(np.random.normal(85, 8), 65, 100)

        elif profile == 1:  # Consistent Average
            gpa = np.clip(np.random.normal(7.3, 0.5), 6.2, 8.4)
            attendance = np.clip(np.random.normal(79, 6), 68, 92)
            study_hours = np.clip(np.random.normal(3.3, 0.8), 2.0, 5.5)
            internal_marks = np.clip(np.random.normal(71, 7), 55, 85)
            prev_sub_marks = np.clip(np.random.normal(70, 8), 52, 86)
            quiz_marks = np.clip(np.random.normal(72, 7), 55, 86)
            assign_marks = np.clip(np.random.normal(76, 7), 60, 90)
            assign_comp = np.clip(np.random.normal(80, 8), 65, 96)
            quiz_part = np.clip(np.random.normal(82, 8), 65, 98)
            learning_act = np.clip(np.random.normal(65, 10), 45, 88)

        elif profile == 2:  # Inconsistent / Moderate Risk
            gpa = np.clip(np.random.normal(6.1, 0.5), 5.0, 7.2)
            attendance = np.clip(np.random.normal(67, 7), 52, 80)
            study_hours = np.clip(np.random.normal(2.1, 0.6), 1.0, 3.8)
            internal_marks = np.clip(np.random.normal(57, 8), 40, 72)
            prev_sub_marks = np.clip(np.random.normal(58, 8), 42, 74)
            quiz_marks = np.clip(np.random.normal(56, 9), 38, 72)
            assign_marks = np.clip(np.random.normal(62, 9), 45, 78)
            assign_comp = np.clip(np.random.normal(64, 10), 45, 82)
            quiz_part = np.clip(np.random.normal(68, 10), 48, 85)
            learning_act = np.clip(np.random.normal(48, 11), 25, 70)

        else:  # Severe Risk / Disengaged
            gpa = np.clip(np.random.normal(4.8, 0.6), 3.0, 6.0)
            attendance = np.clip(np.random.normal(52, 8), 30, 68)
            study_hours = np.clip(np.random.normal(1.2, 0.5), 0.3, 2.5)
            internal_marks = np.clip(np.random.normal(43, 8), 25, 58)
            prev_sub_marks = np.clip(np.random.normal(44, 9), 25, 60)
            quiz_marks = np.clip(np.random.normal(42, 9), 20, 58)
            assign_marks = np.clip(np.random.normal(48, 10), 25, 65)
            assign_comp = np.clip(np.random.normal(46, 12), 20, 65)
            quiz_part = np.clip(np.random.normal(50, 12), 25, 70)
            learning_act = np.clip(np.random.normal(32, 10), 10, 55)

        # Occasional realistic individual variance (e.g. 5% outlier behavior)
        if np.random.rand() < 0.05:
            # High attendance but struggles with exams
            attendance = min(attendance + 15, 100)
            internal_marks = max(internal_marks - 15, 20)
        elif np.random.rand() < 0.05:
            # Low attendance but cramming effectively
            attendance = max(attendance - 15, 35)
            internal_marks = min(internal_marks + 12, 95)

        # Compute Quiz Performance & LBI
        quiz_perf = compute_quiz_performance(quiz_marks, quiz_part)
        lbi = calculate_lbi(
            quiz_performance=quiz_perf,
            assignment_completion=assign_comp,
            attendance=attendance,
            study_hours=study_hours,
            learning_activity=learning_act,
        )

        # Logical formulation of Final Score:
        # Academic factors (70%) + Behavioral factors/LBI (30%) + slight random noise
        # Equivalent breakdown:
        # 0.25 * internal + 0.20 * (GPA*10) + 0.15 * prev_sub + 0.12 * quiz + 0.08 * assign + 0.20 * LBI
        score_base = (
            0.24 * internal_marks
            + 0.18 * (gpa * 10.0)
            + 0.14 * prev_sub_marks
            + 0.12 * quiz_marks
            + 0.08 * assign_marks
            + 0.24 * lbi
        )
        noise = np.random.normal(0, 2.2)
        final_score = np.clip(round(score_base + noise, 2), 15.0, 100.0)

        data.append({
            "student_id": student_id,
            "previous_gpa": round(float(gpa), 2),
            "internal_marks": round(float(internal_marks), 1),
            "previous_subject_marks": round(float(prev_sub_marks), 1),
            "quiz_marks": round(float(quiz_marks), 1),
            "assignment_marks": round(float(assign_marks), 1),
            "attendance": round(float(attendance), 1),
            "study_hours": round(float(study_hours), 2),
            "assignment_completion": round(float(assign_comp), 1),
            "quiz_participation": round(float(quiz_part), 1),
            "learning_activity": round(float(learning_act), 1),
            "lbi": round(float(lbi), 2),
            "final_score": float(final_score),
        })

    df = pd.DataFrame(data)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} records and saved to {output_path}")

    return df


if __name__ == "__main__":
    csv_file = CURRENT_DIR / "student_performance.csv"
    df = generate_student_dataset(n_samples=1500, random_state=42, output_path=csv_file)
    print("\nDataset Summary Preview:")
    print(df.head(5))
    print("\nDataset Descriptive Statistics:")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]])
