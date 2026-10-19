"""
Recommendation and Risk Assessment Module.
Provides rule-based weak area identification, risk level classification,
and dynamic personalized academic recommendations.
"""

from typing import Any, Dict, List, Tuple


# Thresholds for identifying weak areas
THRESHOLDS = {
    "attendance": 75.0,            # Below 75% is considered low attendance
    "study_hours": 2.5,            # Below 2.5 hours/day is considered low study time
    "quiz_performance": 60.0,      # Below 60% is low quiz performance
    "quiz_marks": 60.0,            # Below 60% quiz marks
    "assignment_completion": 70.0, # Below 70% assignment completion rate
    "learning_activity": 50.0,     # Below 50 is low LMS/learning activity
    "internal_marks": 50.0,        # Below 50% internal examination marks
    "previous_gpa": 6.0,           # Below 6.0 CGPA on 10-point scale
}


def classify_risk(predicted_score: float) -> Tuple[str, str, str]:
    """
    Classify student academic risk according to PRD specification:
    - 75–100 -> LOW RISK
    - 60–74  -> MEDIUM RISK
    - < 60   -> HIGH RISK

    Returns:
        (risk_level, color_code, badge_icon)
    """
    score = float(predicted_score)
    if score >= 75.0:
        return "LOW RISK", "#16a34a", "🟢"
    elif score >= 60.0:
        return "MEDIUM RISK", "#d97706", "🟡"
    else:
        return "HIGH RISK", "#dc2626", "🔴"


def identify_weak_areas(inputs: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Evaluate student inputs against established thresholds to detect specific weak areas.
    Returns a list of weak area dictionaries with indicator name, value, threshold, and severity.
    """
    weak_areas = []

    # Check Attendance
    att = inputs.get("attendance", 100.0)
    if att < THRESHOLDS["attendance"]:
        weak_areas.append({
            "key": "attendance",
            "title": "Low Attendance",
            "current_value": f"{att:.1f}%",
            "threshold": f"{THRESHOLDS['attendance']}%",
            "severity": "High" if att < 60.0 else "Medium",
        })

    # Check Daily Study Hours
    study = inputs.get("study_hours", 4.0)
    if study < THRESHOLDS["study_hours"]:
        weak_areas.append({
            "key": "study_hours",
            "title": "Low Study Hours",
            "current_value": f"{study:.1f} hrs/day",
            "threshold": f"{THRESHOLDS['study_hours']} hrs/day",
            "severity": "High" if study < 1.5 else "Medium",
        })

    # Check Quiz Performance / Quiz Marks
    quiz = inputs.get("quiz_marks", 100.0)
    if quiz < THRESHOLDS["quiz_marks"]:
        weak_areas.append({
            "key": "quiz_performance",
            "title": "Low Quiz Performance",
            "current_value": f"{quiz:.1f}%",
            "threshold": f"{THRESHOLDS['quiz_marks']}%",
            "severity": "High" if quiz < 45.0 else "Medium",
        })

    # Check Assignment Completion Rate
    assign = inputs.get("assignment_completion", 100.0)
    if assign < THRESHOLDS["assignment_completion"]:
        weak_areas.append({
            "key": "assignment_completion",
            "title": "Low Assignment Completion",
            "current_value": f"{assign:.1f}%",
            "threshold": f"{THRESHOLDS['assignment_completion']}%",
            "severity": "High" if assign < 50.0 else "Medium",
        })

    # Check Learning Activity / LMS Usage
    activity = inputs.get("learning_activity", 100.0)
    if activity < THRESHOLDS["learning_activity"]:
        weak_areas.append({
            "key": "learning_activity",
            "title": "Low Learning/LMS Activity",
            "current_value": f"{activity:.1f}/100",
            "threshold": f"{THRESHOLDS['learning_activity']}/100",
            "severity": "High" if activity < 35.0 else "Medium",
        })

    # Check Internal Exam Marks
    internal = inputs.get("internal_marks", 100.0)
    if internal < THRESHOLDS["internal_marks"]:
        weak_areas.append({
            "key": "internal_marks",
            "title": "Low Internal Exam Marks",
            "current_value": f"{internal:.1f}%",
            "threshold": f"{THRESHOLDS['internal_marks']}%",
            "severity": "High" if internal < 40.0 else "Medium",
        })

    # Check Previous GPA (if available)
    gpa = inputs.get("previous_gpa", 10.0)
    if gpa < THRESHOLDS["previous_gpa"]:
        weak_areas.append({
            "key": "previous_gpa",
            "title": "Low Previous GPA/CGPA",
            "current_value": f"{gpa:.2f}/10.0",
            "threshold": f"{THRESHOLDS['previous_gpa']}/10.0",
            "severity": "High" if gpa < 5.0 else "Medium",
        })

    return weak_areas


def generate_recommendations(
    inputs: Dict[str, float],
    predicted_score: float,
    weak_areas: List[Dict[str, Any]],
) -> List[str]:
    """
    Generate dynamic, actionable, rule-based recommendations tailored to the student's weaknesses
    and overarching predicted risk tier.
    """
    recommendations = []
    weak_keys = {wa["key"] for wa in weak_areas}

    # Weakness-specific recommendations
    if "attendance" in weak_keys:
        recommendations.append(
            "Improve attendance and maintain regular participation in classes (target at least 75%–85%)."
        )

    if "study_hours" in weak_keys:
        recommendations.append(
            "Increase daily study time and follow a structured study schedule (aim for 3 to 4 hours of focused study daily)."
        )

    if "quiz_performance" in weak_keys:
        recommendations.append(
            "Practice additional quizzes and revise difficult topics prior to upcoming assessments."
        )

    if "assignment_completion" in weak_keys:
        recommendations.append(
            "Complete pending assignments and maintain timely submission to secure full internal assessment credit."
        )

    if "learning_activity" in weak_keys:
        recommendations.append(
            "Increase participation in learning activities and regularly use available LMS learning resources and video lectures."
        )

    if "internal_marks" in weak_keys:
        recommendations.append(
            "Review past internal exam mistakes, clarify doubts with course instructors during office hours, and focus on core technical concepts."
        )

    if "previous_gpa" in weak_keys:
        recommendations.append(
            "Revisit prerequisite course material to strengthen subject fundamentals and improve overall academic cumulative standing."
        )

    # Risk-level specific overarching guidance
    risk_level, _, _ = classify_risk(predicted_score)
    if risk_level == "HIGH RISK":
        recommendations.insert(
            0,
            "⚠️ URGENT INTERVENTION: Schedule an academic counseling session with your faculty mentor or advisor immediately."
        )
    elif risk_level == "MEDIUM RISK":
        if not recommendations:
            recommendations.append(
                "Good progress, but incremental consistency in daily revision will push your performance into the Low Risk tier (>=75%)."
            )
    else:  # LOW RISK
        if not recommendations:
            recommendations.append(
                "🌟 Excellent academic trajectory! Maintain this balanced learning routine and consider taking up advanced problems or mentoring peers."
            )

    return recommendations
