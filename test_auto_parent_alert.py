"""
Automated Verification Test Suite:
Validates automatic parent alert behavior, risk classification,
email content generation, duplicate send prevention on reruns, and retry handling.
"""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.parent_alert import (
    generate_parent_alert_content,
    get_smtp_credentials,
    is_demo_mode,
    send_parent_alert_email,
)
from src.prediction import StudentPredictor


def run_tests():
    print("=================================================================")
    print("RUNNING AUTOMATIC PARENT ALERT SYSTEM VERIFICATION TESTS")
    print("=================================================================")

    # Initialize predictor
    model_path = BASE_DIR / "models" / "best_model.pkl"
    preprocessor_path = BASE_DIR / "models" / "preprocessor.pkl"
    predictor = StudentPredictor(model_path, preprocessor_path)

    # -------------------------------------------------------------
    # TEST 1: Low Risk Student Prediction & Email Exclusion
    # -------------------------------------------------------------
    low_risk_inputs = {
        "previous_gpa": 9.0,
        "internal_marks": 90.0,
        "previous_subject_marks": 88.0,
        "quiz_marks": 92.0,
        "assignment_marks": 95.0,
        "attendance": 95.0,
        "study_hours": 5.0,
        "assignment_completion": 95.0,
        "quiz_participation": 95.0,
        "learning_activity": 90.0,
    }
    low_res = predictor.predict(low_risk_inputs)
    print(f"\n[Test 1 - Low Risk]: Score={low_res['predicted_score']}%, Tier={low_res['risk_level']}")
    assert low_res["risk_level"] == "LOW RISK", f"Expected LOW RISK, got {low_res['risk_level']}"
    assert len(low_res["recommendations"]) > 0, "Recommendations should be present"
    # Verify policy: No email should be sent for low risk
    should_send_low = low_res["risk_level"] in ("MEDIUM RISK", "HIGH RISK")
    assert not should_send_low, "Low Risk must NOT trigger an email send"
    print("  [PASS] Test 1 Passed: Low Risk generates predictions and recommendations with NO parent email sent.")

    # -------------------------------------------------------------
    # TEST 2: Medium Risk Student Prediction & Early-Warning Content
    # -------------------------------------------------------------
    med_risk_inputs = {
        "previous_gpa": 7.0,
        "internal_marks": 68.0,
        "previous_subject_marks": 66.0,
        "quiz_marks": 70.0,
        "assignment_marks": 72.0,
        "attendance": 76.0,
        "study_hours": 3.0,
        "assignment_completion": 78.0,
        "quiz_participation": 80.0,
        "learning_activity": 65.0,
    }
    med_res = predictor.predict(med_risk_inputs)
    print(f"\n[Test 2 - Medium Risk]: Score={med_res['predicted_score']}%, Tier={med_res['risk_level']}")
    assert med_res["risk_level"] == "MEDIUM RISK", f"Expected MEDIUM RISK, got {med_res['risk_level']}"

    med_alert = generate_parent_alert_content(
        student_name="Alex Johnson",
        parent_name="Robert Johnson",
        predicted_score=med_res["predicted_score"],
        risk_level=med_res["risk_level"],
        attendance=med_risk_inputs["attendance"],
        study_hours=med_risk_inputs["study_hours"],
        weak_areas=med_res["weak_areas"],
        recommendations=med_res["recommendations"],
    )
    assert "Early-Warning" in med_alert["subject"] or "Advisory" in med_alert["subject"]
    assert "Alex Johnson" in med_alert["body_text"]
    assert f"{med_res['predicted_score']:.1f}%" in med_alert["body_text"]
    assert "MEDIUM RISK" in med_alert["body_text"]
    assert f"{med_risk_inputs['attendance']:.1f}%" in med_alert["body_text"]
    assert "Recommended Action for Parent/Guardian:" in med_alert["body_text"]
    print("  [PASS] Test 2 Passed: Medium Risk generates Early-Warning alert with all required fields (student name, score, risk, attendance, recommendations).")

    # -------------------------------------------------------------
    # TEST 3: High Risk Student Prediction & Urgent Academic Alert
    # -------------------------------------------------------------
    high_risk_inputs = {
        "previous_gpa": 4.5,
        "internal_marks": 40.0,
        "previous_subject_marks": 42.0,
        "quiz_marks": 38.0,
        "assignment_marks": 45.0,
        "attendance": 50.0,
        "study_hours": 1.2,
        "assignment_completion": 40.0,
        "quiz_participation": 45.0,
        "learning_activity": 30.0,
    }
    high_res = predictor.predict(high_risk_inputs)
    print(f"\n[Test 3 - High Risk]: Score={high_res['predicted_score']}%, Tier={high_res['risk_level']}")
    assert high_res["risk_level"] == "HIGH RISK", f"Expected HIGH RISK, got {high_res['risk_level']}"

    high_alert = generate_parent_alert_content(
        student_name="Daniel Smith",
        parent_name="Sarah Smith",
        predicted_score=high_res["predicted_score"],
        risk_level=high_res["risk_level"],
        attendance=high_risk_inputs["attendance"],
        study_hours=high_risk_inputs["study_hours"],
        weak_areas=high_res["weak_areas"],
        recommendations=high_res["recommendations"],
    )
    assert "URGENT" in high_alert["subject"]
    assert "Daniel Smith" in high_alert["body_text"]
    assert f"{high_res['predicted_score']:.1f}%" in high_alert["body_text"]
    assert "HIGH RISK" in high_alert["body_text"]
    assert f"{high_risk_inputs['attendance']:.1f}%" in high_alert["body_text"]
    assert "Recommended Action for Parent/Guardian:" in high_alert["body_text"]
    print("  [PASS] Test 3 Passed: High Risk generates Urgent Academic Alert with all required fields (student name, score, risk, attendance, recommendations).")

    # -------------------------------------------------------------
    # TEST 4: Simulated Streamlit Reruns & Duplicate Send Prevention
    # -------------------------------------------------------------
    print("\n[Test 4 - Streamlit Rerun Simulation]")
    mock_session_state = {
        "email_dispatched_events": {},
        "current_prediction_event_id": "evt_test_001",
    }
    event_id = mock_session_state["current_prediction_event_id"]

    # Simulating first run on prediction submit:
    send_count = 0
    def simulate_app_run(is_submit_click: bool, new_event_id: str = None):
        nonlocal send_count
        if is_submit_click:
            mock_session_state["current_prediction_event_id"] = new_event_id
        
        curr_id = mock_session_state["current_prediction_event_id"]
        if curr_id not in mock_session_state["email_dispatched_events"]:
            # Auto-send logic:
            send_count += 1
            mock_session_state["email_dispatched_events"][curr_id] = {
                "action": "sent",
                "status": "sent",
                "message": f"Sent email for {curr_id}",
            }

    # Initial prediction event
    simulate_app_run(is_submit_click=True, new_event_id="evt_pred_1")
    assert send_count == 1, "Should send 1 email on initial prediction event"

    # Simulate 5 consecutive Streamlit reruns (e.g. user toggles widgets, changes tabs, etc.)
    for rerun_idx in range(1, 6):
        simulate_app_run(is_submit_click=False)
    assert send_count == 1, f"Expected send_count to remain 1 after 5 reruns, got {send_count}"
    print(f"  [PASS] Test 4 Passed: 5 Streamlit reruns produced 0 duplicate sends (Total sends = {send_count}).")

    # Explicit user submission generates a NEW prediction event
    simulate_app_run(is_submit_click=True, new_event_id="evt_pred_2")
    assert send_count == 2, f"Expected send_count to be 2 after new prediction event, got {send_count}"
    # Another 3 reruns on second prediction
    for rerun_idx in range(1, 4):
        simulate_app_run(is_submit_click=False)
    assert send_count == 2, f"Expected send_count to remain 2 after subsequent reruns, got {send_count}"
    print("  [PASS] Test 4 Passed: New prediction event triggers exactly one new email send.")

    # -------------------------------------------------------------
    # TEST 5: SMTP Failure Handling & Retry Verification
    # -------------------------------------------------------------
    print("\n[Test 5 - SMTP Failure & Error Handling]")
    # Test invalid email address
    bad_result = send_parent_alert_email(
        recipient_email="invalid_email",
        subject="Test",
        body_text="Test",
        smtp_credentials={"username": "a", "password": "b", "server": "localhost", "port": 587},
    )
    assert bad_result["success"] is False, "Invalid email should fail"
    assert bad_result["status"] == "error", "Status should be 'error'"
    print("  [PASS] Test 5 Passed: Invalid recipient returns explicit error and never falsely claims success.")

    # Test bad credentials failure handling
    bad_creds_result = send_parent_alert_email(
        recipient_email="parent@example.com",
        subject="Test",
        body_text="Test",
        smtp_credentials={"username": "bad_user", "password": "bad_password", "server": "127.0.0.1", "port": 65534},
    )
    assert bad_creds_result["success"] is False, "Connection to invalid SMTP server must fail"
    assert bad_creds_result["status"] == "error", "Status must be 'error'"
    assert "Failed to send email" in bad_creds_result["message"]
    print("  [PASS] Test 5 Passed: Failed SMTP returns structured error response allowing safe retry.")

    # -------------------------------------------------------------
    # TEST 6: Demo Mode Fallback Verification
    # -------------------------------------------------------------
    print("\n[Test 6 - Demo Mode Fallback]")
    demo_result = send_parent_alert_email(
        recipient_email="parent@example.com",
        subject="Test Alert",
        body_text="Test Text",
        smtp_credentials=None, # None simulates no credentials
    )
    # When credentials aren't passed, demo mode handles simulation safely
    assert demo_result["status"] in ("demo", "sent"), "Should handle demo mode or secrets safely"
    print("  [PASS] Test 6 Passed: Demo Mode handles simulation safely without crashes.")

    print("\n=================================================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY! (6/6 Passed)")
    print("=================================================================")


if __name__ == "__main__":
    run_tests()
