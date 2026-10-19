"""
Streamlit Web Application:
AI-Based Student Performance Prediction and Early Warning System Using Machine Learning
Page Title: Student Performance Predictor
"""

import json
from pathlib import Path
import sys
import uuid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup paths and imports
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.feature_engineering import calculate_lbi, compute_quiz_performance
from src.parent_alert import (
    generate_parent_alert_content,
    is_demo_mode,
    send_parent_alert_email,
)
from src.prediction import StudentPredictor


def get_secrets_smtp_credentials():
    """
    Safely retrieve Gmail SMTP credentials from st.secrets without hardcoding.
    Reads: EMAIL_SENDER, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT.
    Never logs or exposes the password. Returns credentials dict or None for Demo Mode.
    """
    try:
        if hasattr(st, "secrets"):
            sender = st.secrets.get("EMAIL_SENDER") or (st.secrets.get("smtp", {}).get("sender") if "smtp" in st.secrets else None)
            password = st.secrets.get("EMAIL_PASSWORD") or (st.secrets.get("smtp", {}).get("password") if "smtp" in st.secrets else None)
            server = st.secrets.get("SMTP_SERVER") or (st.secrets.get("smtp", {}).get("server") if "smtp" in st.secrets else "smtp.gmail.com")
            port = st.secrets.get("SMTP_PORT") or (st.secrets.get("smtp", {}).get("port") if "smtp" in st.secrets else 587)

            if sender and password:
                return {
                    "server": str(server).strip(),
                    "port": int(port),
                    "username": str(sender).strip(),
                    "password": str(password).strip(),
                    "sender": str(sender).strip(),
                }
    except Exception:
        pass
    return None



# Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #111827;
        margin: 0.3rem 0;
    }
    .risk-badge {
        display: inline-block;
        font-size: 1.25rem;
        font-weight: 700;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        color: white;
    }
    .weak-tag {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.9rem;
        margin: 0.25rem 0;
        font-weight: 500;
    }
    .rec-item {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 0.65rem 0.9rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        color: #14532d;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 600;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    """Load model, preprocessor and cached metrics."""
    model_path = BASE_DIR / "models" / "best_model.pkl"
    preprocessor_path = BASE_DIR / "models" / "preprocessor.pkl"
    metrics_path = BASE_DIR / "models" / "model_metrics.json"
    

    if not model_path.exists() or not preprocessor_path.exists():
        st.error(
            "Trained model artifacts not found. Please run `python train_model.py` first."
        )
        return None, None

    predictor = StudentPredictor(model_path, preprocessor_path)

    metrics = None
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return predictor, metrics


predictor, metrics_summary = load_predictor()


# ---------------- SIDEBAR: QUICK PRESET SELECTION ----------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/graduation-cap.png",
        width=70,
    )
    st.header("⚡ Quick Preset Profiles")
    st.write("Select a pre-filled student archetype to test different scenarios:")

    preset = st.radio(
        "Choose Profile:",
        ["Custom Input", "High Achiever (Low Risk)", "Average Student (Medium Risk)", "At-Risk Student (High Risk)"],
        index=0,
    )

    if preset == "High Achiever (Low Risk)":
        def_stu_name = "Emily Davis"
        def_parent_name = "Michael Davis"
        def_parent_email = "michael.davis@example.com"
        def_parent_phone = "+1 (555) 234-5678"
        def_gpa, def_internal, def_prev_sub, def_quiz, def_assign = 8.8, 88.0, 85.0, 90.0, 92.0
        def_att, def_hours, def_comp, def_part, def_act = 92.0, 5.0, 95.0, 95.0, 88.0
    elif preset == "Average Student (Medium Risk)":
        def_stu_name = "Alex Johnson"
        def_parent_name = "Robert Johnson"
        def_parent_email = "robert.johnson@example.com"
        def_parent_phone = "+1 (555) 345-6789"
        def_gpa, def_internal, def_prev_sub, def_quiz, def_assign = 7.2, 68.0, 67.0, 70.0, 75.0
        def_att, def_hours, def_comp, def_part, def_act = 78.0, 3.0, 80.0, 80.0, 65.0
    elif preset == "At-Risk Student (High Risk)":
        def_stu_name = "Daniel Smith"
        def_parent_name = "Sarah Smith"
        def_parent_email = "sarah.smith@example.com"
        def_parent_phone = "+1 (555) 456-7890"
        def_gpa, def_internal, def_prev_sub, def_quiz, def_assign = 4.8, 42.0, 45.0, 40.0, 48.0
        def_att, def_hours, def_comp, def_part, def_act = 55.0, 1.2, 45.0, 50.0, 30.0
    else:
        # Default standard values
        def_stu_name = "Alex Johnson"
        def_parent_name = "Robert Johnson"
        def_parent_email = "parent@example.com"
        def_parent_phone = "+1 (555) 012-3456"
        def_gpa, def_internal, def_prev_sub, def_quiz, def_assign = 7.5, 70.0, 68.0, 72.0, 75.0
        def_att, def_hours, def_comp, def_part, def_act = 80.0, 3.0, 85.0, 85.0, 60.0

    st.markdown("---")
    st.markdown(
        """
        **System Information**
        - **Target:** Final Score (0–100%)
        - **Algorithm:** ML Regression
        - **Metric Standards & Policy:**
          - **75–100% : 🟢 LOW RISK**
            *Satisfactory progress; no parent alert required.*
          - **60–74%  : 🟡 MEDIUM RISK**
            *Early-warning parent alert (performance requires attention).*
          - **< 60%   : 🔴 HIGH RISK**
            *Urgent parent alert (immediate intervention recommended).*
        """
    )


# ---------------- MAIN CONTENT ----------------
st.markdown('<div class="main-title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-Based Student Performance Prediction and Early Warning System Using Machine Learning</div>',
    unsafe_allow_html=True,
)

tab_prediction, tab_models, tab_about = st.tabs(
    ["📊 Student Prediction Dashboard", "📈 Model Performance & Comparison", "ℹ️ About & Methodology"]
)


# =========================================================================
# TAB 1: PREDICTION DASHBOARD
# =========================================================================
with tab_prediction:
    st.subheader("1. Enter Student Details")
    st.caption("Provide academic records and learning engagement behaviors for evaluation.")

    with st.form("student_input_form"):
        # Student & Parent Contact Information
        st.markdown("#### 👤 Student & Parent / Guardian Profile")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            student_name = st.text_input(
                "Student Name",
                value=def_stu_name,
                help="Registered student name.",
            )
        with col_p2:
            parent_name = st.text_input(
                "Parent / Guardian Name",
                value=def_parent_name,
                help="Name of parent or primary legal guardian.",
            )
        with col_p3:
            parent_email = st.text_input(
                "Parent Email Address",
                value=def_parent_email,
                help="Parent/guardian email to receive academic alerts.",
            )
        with col_p4:
            parent_phone = st.text_input(
                "Parent Mobile Number",
                value=def_parent_phone,
                help="Parent contact phone number (optional).",
            )

        st.markdown("---")
        col_acad, col_behav = st.columns(2, gap="large")

        # Academic Factors
        with col_acad:
            st.markdown("#### 📚 Academic Factors")


            gpa = st.number_input(
                "Previous GPA / CGPA (Scale 0.0 - 10.0)",
                min_value=0.0,
                max_value=10.0,
                value=float(def_gpa),
                step=0.1,
                help="Cumulative Grade Point Average from previous academic semesters (0.0 to 10.0).",
            )

            internal_marks = st.slider(
                "Internal Examination Marks (0 - 100)",
                min_value=0.0,
                max_value=100.0,
                value=float(def_internal),
                step=0.5,
                help="Mid-term / internal exam score percentage.",
            )

            prev_subject_marks = st.slider(
                "Previous Subject Marks (0 - 100)",
                min_value=0.0,
                max_value=100.0,
                value=float(def_prev_sub),
                step=0.5,
                help="Marks obtained in prerequisite/related subjects.",
            )

            quiz_marks = st.slider(
                "Quiz Marks Average (0 - 100)",
                min_value=0.0,
                max_value=100.0,
                value=float(def_quiz),
                step=0.5,
                help="Average marks obtained in formative in-class quizzes.",
            )

            assignment_marks = st.slider(
                "Assignment Marks Average (0 - 100)",
                min_value=0.0,
                max_value=100.0,
                value=float(def_assign),
                step=0.5,
                help="Evaluation score on submitted home assignments and coursework.",
            )

        # Learning Behavior Factors
        with col_behav:
            st.markdown("#### 🧠 Learning Behavior Factors")

            attendance = st.slider(
                "Attendance Percentage (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(def_att),
                step=0.5,
                help="Classroom attendance percentage for the current semester.",
            )

            study_hours = st.slider(
                "Study Hours Per Day (Hours)",
                min_value=0.0,
                max_value=12.0,
                value=float(def_hours),
                step=0.25,
                help="Dedicated self-study time per day outside of lectures.",
            )

            assignment_completion = st.slider(
                "Assignment Completion Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(def_comp),
                step=0.5,
                help="Percentage of assigned coursework submitted on time.",
            )

            quiz_participation = st.slider(
                "Quiz Participation Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(def_part),
                step=0.5,
                help="Ratio of completed quizzes relative to total assigned quizzes.",
            )

            learning_activity = st.slider(
                "Learning / LMS Activity Level (0 - 100)",
                min_value=0.0,
                max_value=100.0,
                value=float(def_act),
                step=1.0,
                help="Overall engagement index on the learning management system (course videos, reading materials, forums).",
            )

        submit_btn = st.form_submit_button(
            "🔮 Predict Performance",
            use_container_width=True,
            type="primary",
        )

    # Prediction Execution & Results Section
    if "email_dispatched_events" not in st.session_state:
        st.session_state["email_dispatched_events"] = {}

    if submit_btn or "has_predicted" in st.session_state:
        if submit_btn:
            # Explicit user submission generates a new unique prediction event
            new_event_id = f"evt_{uuid.uuid4().hex[:10]}"
            st.session_state["has_predicted"] = True
            st.session_state["current_prediction_event_id"] = new_event_id
            st.session_state["student_info"] = {
                "student_name": student_name,
                "parent_name": parent_name,
                "parent_email": parent_email,
                "parent_phone": parent_phone,
            }
            st.session_state["raw_inputs"] = {
                "previous_gpa": gpa,
                "internal_marks": internal_marks,
                "previous_subject_marks": prev_subject_marks,
                "quiz_marks": quiz_marks,
                "assignment_marks": assignment_marks,
                "attendance": attendance,
                "study_hours": study_hours,
                "assignment_completion": assignment_completion,
                "quiz_participation": quiz_participation,
                "learning_activity": learning_activity,
            }

        current_event_id = st.session_state.get("current_prediction_event_id")
        if not current_event_id:
            current_event_id = f"evt_{uuid.uuid4().hex[:10]}"
            st.session_state["current_prediction_event_id"] = current_event_id

        current_stu_info = st.session_state.get(
            "student_info",
            {
                "student_name": student_name,
                "parent_name": parent_name,
                "parent_email": parent_email,
                "parent_phone": parent_phone,
            },
        )
        raw_inputs = st.session_state.get(
            "raw_inputs",
            {
                "previous_gpa": gpa,
                "internal_marks": internal_marks,
                "previous_subject_marks": prev_subject_marks,
                "quiz_marks": quiz_marks,
                "assignment_marks": assignment_marks,
                "attendance": attendance,
                "study_hours": study_hours,
                "assignment_completion": assignment_completion,
                "quiz_participation": quiz_participation,
                "learning_activity": learning_activity,
            },
        )

        if predictor is not None:
            results = predictor.predict(raw_inputs)

            # AUTOMATIC PARENT ALERT DISPATCH
            # Triggers exactly once per prediction event; prevents duplicate sending on reruns
            if current_event_id not in st.session_state["email_dispatched_events"]:
                risk_tier = results["risk_level"]
                auto_stu_name = current_stu_info.get("student_name", "Student")
                auto_par_name = current_stu_info.get("parent_name", "Parent / Guardian")
                auto_par_email = current_stu_info.get("parent_email", "").strip()
                auto_att = float(raw_inputs.get("attendance", 0.0))
                auto_study = float(raw_inputs.get("study_hours", 0.0))

                if risk_tier == "LOW RISK":
                    # Low Risk: Do not send parent email
                    st.session_state["email_dispatched_events"][current_event_id] = {
                        "action": "none",
                        "risk_level": "LOW RISK",
                        "status": "not_required",
                        "message": "Student is performing satisfactorily at Low Risk. No parent email alert required or dispatched.",
                        "recipient": auto_par_email,
                    }
                elif risk_tier in ("MEDIUM RISK", "HIGH RISK"):
                    # Medium Risk or High Risk: Automatically send parent alert email
                    if not auto_par_email or "@" not in auto_par_email:
                        st.session_state["email_dispatched_events"][current_event_id] = {
                            "action": "error",
                            "risk_level": risk_tier,
                            "success": False,
                            "status": "error",
                            "message": "Automatic email dispatch could not proceed: Parent email address is missing or invalid. Please update the email and click Retry.",
                            "recipient": auto_par_email,
                        }
                    else:
                        alert_content_auto = generate_parent_alert_content(
                            student_name=auto_stu_name,
                            parent_name=auto_par_name,
                            predicted_score=results["predicted_score"],
                            risk_level=risk_tier,
                            attendance=auto_att,
                            study_hours=auto_study,
                            weak_areas=results["weak_areas"],
                            recommendations=results["recommendations"],
                        )
                        smtp_creds_auto = get_secrets_smtp_credentials()
                        dispatch_res = send_parent_alert_email(
                            recipient_email=auto_par_email,
                            subject=alert_content_auto["subject"],
                            body_text=alert_content_auto["body_text"],
                            body_html=alert_content_auto["body_html"],
                            smtp_credentials=smtp_creds_auto,
                        )
                        dispatch_res["risk_level"] = risk_tier
                        dispatch_res["alert_type"] = (
                            "Parent Urgent Academic Alert" if risk_tier == "HIGH RISK" else "Parent Early-Warning Advisory"
                        )
                        st.session_state["email_dispatched_events"][current_event_id] = dispatch_res

            st.markdown("---")
            st.subheader("2. Prediction & Risk Assessment Results")

            # Metrics Row: Final Score, LBI, Risk Level
            m_col1, m_col2, m_col3 = st.columns(3)

            with m_col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Predicted Final Score</div>
                        <div class="metric-val" style="color: #2563eb;">{results['predicted_score']:.1f}%</div>
                        <span style="font-size: 0.9rem; color: #6b7280;">Target Scale: 0–100%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m_col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Learning Behavior Index (LBI)</div>
                        <div class="metric-val" style="color: #7c3aed;">{results['lbi']:.1f}</div>
                        <span style="font-size: 0.9rem; color: #6b7280;">Scale: 0–100 (Composite Index)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m_col3:
                risk_style = (
                    "background-color: #16a34a;"
                    if results["risk_level"] == "LOW RISK"
                    else (
                        "background-color: #d97706;"
                        if results["risk_level"] == "MEDIUM RISK"
                        else "background-color: #dc2626;"
                    )
                )
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Academic Risk Tier</div>
                        <div style="margin: 0.6rem 0;">
                            <span class="risk-badge" style="{risk_style}">
                                {results['risk_level']}
                            </span>
                        </div>
                        <span style="font-size: 0.85rem; color: #4b5563;">
                            Threshold: &ge;75% Low | 60–74% Med | &lt;60% High
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # LBI Breakdown Progress Bar
            st.markdown("##### 📌 LBI Component Analysis")
            st.progress(min(max(int(results["lbi"]), 0), 100))
            st.caption(
                f"LBI is calculated as: `0.30×Quiz ({results['input_features']['quiz_marks']:.1f}%) + "
                f"0.25×Assignment ({results['input_features']['assignment_completion']:.1f}%) + "
                f"0.20×Attendance ({results['input_features']['attendance']:.1f}%) + "
                f"0.15×Study Hours ({results['input_features']['study_hours']:.1f} hrs) + "
                f"0.10×LMS Activity ({results['input_features']['learning_activity']:.1f})`"
            )

            # Detailed Weak Areas & Recommendations Row
            w_col, r_col = st.columns(2, gap="medium")

            with w_col:
                st.markdown("#### ⚠️ Identified Weak Areas")
                if results["weak_areas"]:
                    st.write(
                        f"Detected **{len(results['weak_areas'])}** indicator(s) requiring improvement:"
                    )
                    for wa in results["weak_areas"]:
                        severity_color = "#dc2626" if wa["severity"] == "High" else "#d97706"
                        st.markdown(
                            f"""
                            <div class="weak-tag">
                                <strong>• {wa['title']}</strong>: Currently <b>{wa['current_value']}</b> 
                                (Standard benchmark: &ge; {wa['threshold']}) 
                                <span style="float: right; color: {severity_color}; font-weight: 700;">[{wa['severity']} Concern]</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.success(
                        "✅ No weak areas detected! All academic and learning behavior indicators are above benchmark thresholds."
                    )

            with r_col:
                st.markdown("#### 🎯 Personalized Recommendations")
                if results["recommendations"]:
                    for rec in results["recommendations"]:
                        st.markdown(
                            f'<div class="rec-item">✓ {rec}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No immediate interventions needed. Keep up the consistent effort!")

            # =============================================================
            # SECTION 3: PARENT ALERT SYSTEM
            # =============================================================
            st.markdown("---")
            st.subheader("3. 📧 Parent Alert System")
            st.caption(
                "Multi-tiered communication protocol: Provides early-warning alerts to parents/guardians for medium-risk students and urgent alerts for high-risk students to facilitate timely intervention."
            )

            stu_name = current_stu_info.get("student_name", "Student")
            par_name = current_stu_info.get("parent_name", "Parent / Guardian")
            par_email = current_stu_info.get("parent_email", "").strip()
            par_phone = current_stu_info.get("parent_phone", "").strip()
            current_attendance = float(raw_inputs.get("attendance", 0.0))
            current_study_hours = float(raw_inputs.get("study_hours", 0.0))

            if results["risk_level"] == "LOW RISK":
                st.success("✅ **Satisfactory Performance:** Student is currently performing satisfactorily at low academic risk. No parent alert email required or sent.")
                st.markdown(
                    f"""
                    <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-left: 5px solid #16a34a; padding: 0.85rem 1.1rem; border-radius: 6px; margin-top: 0.5rem; color: #166534; font-size: 0.95rem;">
                        <strong>Academic Standing:</strong> {stu_name} is performing satisfactorily across continuous assessments and learning behaviors (Score: {results['predicted_score']:.1f}%). Routine academic tracking continues without parent notification.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:  # MEDIUM RISK or HIGH RISK
                is_high_risk = (results["risk_level"] == "HIGH RISK")
                theme_color = "#dc2626" if is_high_risk else "#d97706"
                badge_label = "HIGH RISK" if is_high_risk else "MEDIUM RISK"

                # Read Gmail SMTP credentials directly from st.secrets without displaying server details in UI
                smtp_creds = get_secrets_smtp_credentials()

                # Generate standardized alert content with differentiated tone and recommendations
                alert_content = generate_parent_alert_content(
                    student_name=stu_name,
                    parent_name=par_name,
                    predicted_score=results["predicted_score"],
                    risk_level=results["risk_level"],
                    attendance=current_attendance,
                    study_hours=current_study_hours,
                    weak_areas=results["weak_areas"],
                    recommendations=results["recommendations"],
                )

                # Small, unobtrusive dispatch status indicator
                dispatch_record = st.session_state["email_dispatched_events"].get(current_event_id, {})
                status_type = dispatch_record.get("status")

                if status_type in ("sent", "demo"):
                    st.markdown(
                        '<div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; '
                        'background-color: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.28); '
                        'border-radius: 6px; color: #22c55e; font-size: 0.88rem; font-weight: 500; margin-bottom: 12px;">'
                        '<span>✓</span> <span>Parent alert sent successfully</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                elif status_type == "error":
                    st.error(
                        f"❌ Failed to send parent alert: {dispatch_record.get('message', 'Delivery error.')}"
                    )

                    # Safe Retry Option
                    st.markdown("##### 🔁 Safe Retry")
                    col_r1, col_r2 = st.columns([1, 2])
                    with col_r1:
                        retry_target_email = st.text_input(
                            "Parent Email Address for Retry",
                            value=par_email,
                            key=f"retry_email_input_{current_event_id}",
                            help="Update or confirm parent email address to retry sending alert.",
                        )
                        retry_clicked = st.button(
                            "🔄 Retry Sending Parent Alert",
                            type="primary",
                            use_container_width=True,
                            key=f"btn_retry_{current_event_id}",
                        )
                    with col_r2:
                        st.write("")
                        st.caption("Automatic delivery encountered an issue. Correct details and click Retry.")

                    if retry_clicked:
                        if not retry_target_email or "@" not in retry_target_email:
                            st.error("⚠️ Please specify a valid parent email address before retrying.")
                        else:
                            with st.spinner("Retrying parent alert dispatch..."):
                                retry_result = send_parent_alert_email(
                                    recipient_email=retry_target_email.strip(),
                                    subject=alert_content["subject"],
                                    body_text=alert_content["body_text"],
                                    body_html=alert_content["body_html"],
                                    smtp_credentials=smtp_creds,
                                )
                                retry_result["risk_level"] = results["risk_level"]
                                retry_result["alert_type"] = (
                                    "Parent Urgent Academic Alert" if is_high_risk else "Parent Early-Warning Advisory"
                                )
                                st.session_state["email_dispatched_events"][current_event_id] = retry_result
                                st.rerun()

                # Layout: Parent Alert Summary
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                        <span style="font-size: 1.15rem; font-weight: 700; color: inherit;">👤 Parent Alert Summary</span>
                        <span style="background-color: {theme_color}; color: #ffffff; padding: 2px 9px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.03em;">
                            {badge_label}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    - **Student Name:** `{stu_name}`
                    - **Parent / Guardian:** `{par_name}`
                    - **Parent Email:** `{par_email or 'Not specified'}`
                    - **Parent Mobile:** `{par_phone or 'Not specified'}`
                    - **Predicted Score:** <span style="color: {theme_color}; font-weight: 700;">{results['predicted_score']:.1f}%</span>
                    - **Academic Risk:** <span style="background-color: {theme_color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{results['risk_level']}</span>
                    - **Class Attendance:** `{current_attendance:.1f}%`
                    - **Daily Study Time:** `{current_study_hours:.1f} hrs/day`
                    - **Guidance for Parent:** {alert_content['parent_recommendation']}
                    """,
                    unsafe_allow_html=True,
                )

                # Optional Resend copy option if already sent successfully
                if status_type in ("sent", "demo"):
                    with st.expander("✉️ Resend Alert Copy to Different Email", expanded=False):
                        resend_col1, resend_col2 = st.columns([2, 1])
                        with resend_col1:
                            copy_email = st.text_input(
                                "Alternative / Additional Recipient Email",
                                value=par_email,
                                key=f"copy_email_{current_event_id}",
                            )
                        with resend_col2:
                            st.write("")
                            send_copy_btn = st.button("📨 Send Copy", key=f"btn_copy_{current_event_id}")

                        if send_copy_btn:
                            if not copy_email or "@" not in copy_email:
                                st.error("⚠️ Please specify a valid email address.")
                            else:
                                with st.spinner("Dispatching alert copy..."):
                                    copy_res = send_parent_alert_email(
                                        recipient_email=copy_email.strip(),
                                        subject=alert_content["subject"],
                                        body_text=alert_content["body_text"],
                                        body_html=alert_content["body_html"],
                                        smtp_credentials=smtp_creds,
                                    )
                                    if copy_res.get("status") == "sent":
                                        st.success(f"✅ {copy_res.get('message')}")
                                    elif copy_res.get("status") == "demo":
                                        st.info(f"🚀 **[Demo Mode Simulation]** {copy_res.get('message')}")
                                    else:
                                        st.error(f"❌ {copy_res.get('message')}")




# =========================================================================
# TAB 2: MODEL PERFORMANCE & COMPARISON
# =========================================================================
with tab_models:
    st.subheader("Model Evaluation and Benchmark Comparison")
    st.write(
        "Four machine learning regression models were trained and benchmarked on independent test data "
        "using standard evaluation metrics: **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, "
        "and **Coefficient of Determination (R² Score)**."
    )

    if metrics_summary:
        comp_data = metrics_summary.get("comparison", {})
        comp_df = pd.DataFrame(comp_data).T

        # Format comparison table
        comp_display = comp_df.copy()
        comp_display["MAE"] = comp_display["MAE"].apply(lambda v: f"{v:.4f}")
        comp_display["RMSE"] = comp_display["RMSE"].apply(lambda v: f"{v:.4f}")
        comp_display["R2"] = comp_display["R2"].apply(lambda v: f"{v:.4f}")

        # Show Table
        st.markdown("#### 📋 Model Performance Metrics Table")
        st.dataframe(comp_display, use_container_width=True)

        best_name = metrics_summary.get("best_model_name", "Linear Regression")
        st.success(
            f"🏆 **Selected Production Model:** **{best_name}** "
            f"(R²: {comp_df.loc[best_name, 'R2']:.4f}, MAE: {comp_df.loc[best_name, 'MAE']:.4f}, "
            f"RMSE: {comp_df.loc[best_name, 'RMSE']:.4f})"
        )

        st.markdown("---")
        st.markdown("#### 📊 Comparative Metric Visualizations")

        # 3 Side-by-side or styled charts
        chart_col1, chart_col2, chart_col3 = st.columns(3)

        models_list = list(comp_df.index)
        palette = sns.color_palette("viridis", len(models_list))

        # Chart 1: R2 Score (Higher is better)
        with chart_col1:
            fig1, ax1 = plt.subplots(figsize=(4.5, 3.8))
            bars1 = ax1.bar(
                models_list,
                comp_df["R2"],
                color="#2563eb",
                alpha=0.85,
                edgecolor="#1e3a8a",
                width=0.55,
            )
            ax1.set_title("R² Score (Higher is Better)", fontsize=11, fontweight="bold")
            ax1.set_ylabel("R² Score", fontsize=10)
            ax1.set_ylim(0.85, 1.0)
            ax1.set_xticklabels(models_list, rotation=35, ha="right", fontsize=8.5)
            ax1.grid(axis="y", linestyle="--", alpha=0.5)
            for bar in bars1:
                yval = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + 0.005,
                    f"{yval:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
            st.pyplot(fig1)

        # Chart 2: MAE (Lower is better)
        with chart_col2:
            fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
            bars2 = ax2.bar(
                models_list,
                comp_df["MAE"],
                color="#059669",
                alpha=0.85,
                edgecolor="#065f46",
                width=0.55,
            )
            ax2.set_title("MAE (Lower is Better)", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Mean Absolute Error", fontsize=10)
            ax2.set_ylim(0, max(comp_df["MAE"]) * 1.25)
            ax2.set_xticklabels(models_list, rotation=35, ha="right", fontsize=8.5)
            ax2.grid(axis="y", linestyle="--", alpha=0.5)
            for bar in bars2:
                yval = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + 0.08,
                    f"{yval:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
            st.pyplot(fig2)

        # Chart 3: RMSE (Lower is better)
        with chart_col3:
            fig3, ax3 = plt.subplots(figsize=(4.5, 3.8))
            bars3 = ax3.bar(
                models_list,
                comp_df["RMSE"],
                color="#d97706",
                alpha=0.85,
                edgecolor="#92400e",
                width=0.55,
            )
            ax3.set_title("RMSE (Lower is Better)", fontsize=11, fontweight="bold")
            ax3.set_ylabel("Root Mean Squared Error", fontsize=10)
            ax3.set_ylim(0, max(comp_df["RMSE"]) * 1.25)
            ax3.set_xticklabels(models_list, rotation=35, ha="right", fontsize=8.5)
            ax3.grid(axis="y", linestyle="--", alpha=0.5)
            for bar in bars3:
                yval = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + 0.1,
                    f"{yval:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
            st.pyplot(fig3)

        # Dataset split statistics
        st.markdown("---")
        st.markdown("#### 📂 Dataset & Split Information")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Total Records", 1500)
        col_s2.metric("Train Split", f"{metrics_summary.get('train_samples', 1200)} (80%)")
        col_s3.metric("Test Split", f"{metrics_summary.get('test_samples', 300)} (20%)")
        col_s4.metric("Input Features", len(metrics_summary.get("feature_columns", [])))

    else:
        st.warning("Model comparison metrics file (`models/model_metrics.json`) not found.")


# =========================================================================
# TAB 3: ABOUT & METHODOLOGY
# =========================================================================
with tab_about:
    st.subheader("About the System & Theoretical Framework")
    st.markdown(
        """
        ### 1. Problem Statement
        Traditional higher education academic assessments evaluate students retrospectively at the end of semesters, 
        preventing instructors and students from acting early on learning gaps.
        This system combines **prior academic history** with **continuous learning behavior metrics** to 
        forecast final examination scores and prescribe personalized corrective actions mid-semester.

        ### 2. Learning Behavior Index (LBI) Formulation
        The composite Learning Behavior Index (0–100) measures daily commitment and academic engagement:
        
        $$\\text{LBI} = 0.30 \\times \\text{Quiz Performance} + 0.25 \\times \\text{Assignment Completion} + 0.20 \\times \\text{Attendance} + 0.15 \\times \\text{Study Hours}_{\\text{norm}} + 0.10 \\times \\text{LMS Activity}_{\\text{norm}}$$

        - Daily study hours are normalized against a 10-hour benchmark.
        - Learning/LMS activity is bounded within a 0–100 percentile index.

        ### 3. Academic Risk Classification
        - **🟢 LOW RISK (75% – 100%):** On track for first-class distinction or strong performance.
        - **🟡 MEDIUM RISK (60% – 74%):** Satisfactory performance with scope for targeted improvement.
        - **🔴 HIGH RISK (< 60%):** Critical early warning indicating urgent intervention required.

        ### 4. Machine Learning Pipeline Architecture
        1. **Data Ingestion:** Collection of academic indicators and LMS behavioral logs.
        2. **Preprocessing:** Duplicate removal, median imputation, and standard scaling.
        3. **Regression Benchmark:** Linear Regression, Decision Tree, Random Forest, and SVR.
        4. **Rule-Based Recommendation Engine:** Identifies weak areas below institutional thresholds and generates specific action points.
        5. **Parent Alert System:** Evaluates predicted risk tier to trigger targeted early warning alerts for parents/guardians.

        ### 5. Parent Alert System Protocol
        - **🟢 LOW RISK (&ge; 75%):** Satisfactory academic standing. No parent alert required.
        - **🟡 MEDIUM RISK (60% – 74%):** Polite early-warning parent alert automatically dispatched to inform parents/guardians that the student's performance requires attention, promoting timely supportive intervention without framing it as an emergency.
        - **🔴 HIGH RISK (< 60%):** Strong urgent academic alert automatically dispatched indicating that immediate academic attention, mentor counseling, and structured parental support are recommended.
        - **Automatic Dispatch & Duplicate Prevention:** Alerts are dispatched automatically upon prediction completion without requiring manual button clicks. Session-state event tracking ensures alerts are dispatched strictly once per prediction event and never repeated on ordinary Streamlit reruns. Safe retry is provided in case of transmission failures.
        - **Security & Delivery:** Supports live SMTP email dispatch (configured securely via Streamlit secrets or environment variables) as well as a safe Demo Mode for offline review and viva evaluations.
        """
    )

