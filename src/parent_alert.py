"""
Parent Alert System Module.
Generates structured early-warning alert notifications for parents/guardians
of at-risk students and handles safe email dispatch with Demo Mode fallback.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple


def get_parent_recommendation(
    risk_level: str,
    attendance: float,
    study_hours: float,
    weak_areas: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Generate a concise, constructive, actionable recommendation tailored for parents.
    Adapts tone based on whether the student is at Medium Risk (advisory) or High Risk (urgent).
    """
    recommendations = []

    if attendance < 75.0:
        recommendations.append(
            f"Attendance is currently at {attendance:.1f}% (below the 75% institutional benchmark). Please ensure regular classroom attendance."
        )

    if study_hours < 2.5:
        recommendations.append(
            f"Daily study time is {study_hours:.1f} hrs/day. Encourage dedicated 3 to 4 hours of quiet, focused study at home."
        )

    if weak_areas:
        weak_keys = {wa.get("key") for wa in weak_areas if isinstance(wa, dict)}
        if "assignment_completion" in weak_keys:
            recommendations.append("Assist in monitoring timely completion and submission of course assignments.")
        if "quiz_performance" in weak_keys or "quiz_marks" in weak_keys:
            recommendations.append("Encourage regular revision and topic practice for ongoing formative quizzes.")

    if not recommendations:
        if risk_level == "HIGH RISK":
            recommendations.append(
                "Immediate academic counseling is recommended. Please contact the department faculty advisor or course mentor."
            )
        else:
            recommendations.append(
                "Maintain encouraging oversight on daily study routines, review progress in upcoming assessments, and consult course instructors if concepts need clarification."
            )

    return " ".join(recommendations)


def generate_parent_alert_content(
    student_name: str,
    parent_name: str,
    predicted_score: float,
    risk_level: str,
    attendance: float,
    study_hours: float = 3.0,
    weak_areas: Optional[List[Dict[str, Any]]] = None,
    recommendations: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Construct standardized, professional parent alert email subject, plain text, and HTML body.
    - MEDIUM RISK: Polite early-warning advisory informing that performance requires attention.
    - HIGH RISK: Strong urgent academic alert indicating immediate intervention is recommended.
    Includes student name, predicted score, risk level, attendance, daily study hours, recommendations, and parent guidance.
    """
    parent_salutation = parent_name.strip() if parent_name.strip() else "Parent / Guardian"
    student_display = student_name.strip() if student_name.strip() else "Student"
    parent_rec = get_parent_recommendation(risk_level, attendance, study_hours, weak_areas)

    if risk_level == "HIGH RISK":
        subject = f"URGENT: Student Performance Academic Alert - {student_display}"
        opening_statement = "This is an urgent early warning regarding your ward's projected academic performance."
        core_message = "The system has identified critical academic risks that require immediate academic attention and faculty counseling."
        closing_statement = "Please treat this notification with priority, monitor attendance and study routines closely, and schedule a consultation with the academic mentor."
        header_title = "🚨 Urgent Academic Early Warning Alert"
        accent_color = "#dc2626"
        banner_bg = "#fef2f2"
        banner_border = "#ef4444"
        banner_text_color = "#991b1b"
        alert_tier_badge = "CRITICAL / HIGH RISK"
    else:  # MEDIUM RISK (or polite advisory)
        subject = f"Student Performance Early-Warning Advisory - {student_display}"
        opening_statement = "This is a polite early-warning academic progress update regarding your ward's performance."
        core_message = "The system has identified that the student's performance requires attention to ensure steady progress toward examination milestones."
        closing_statement = "Please encourage regular study routines and discuss progress with your ward to help them improve before final examinations."
        header_title = "🎓 Academic Early-Warning Progress Advisory"
        accent_color = "#d97706"
        banner_bg = "#fffbeb"
        banner_border = "#f59e0b"
        banner_text_color = "#92400e"
        alert_tier_badge = "MODERATE / MEDIUM RISK"

    # Plain text format recommendations block
    rec_text_block = ""
    if recommendations:
        formatted_recs = "\n".join(f"  • {r}" for r in recommendations)
        rec_text_block = f"\nAcademic Recommendations for Student:\n{formatted_recs}\n"

    # Plain text format matching requirements
    body_text = f"""Dear {parent_salutation},

{opening_statement}

Student Name: {student_display}
Predicted Performance: {predicted_score:.1f}%
Risk Level: {risk_level}
Attendance: {attendance:.1f}%
Daily Study Hours: {study_hours:.1f} hrs/day
{rec_text_block}
{core_message}

Recommended Action for Parent/Guardian:
{parent_rec}

{closing_statement}

Regards,
Student Performance Prediction System
Academic Early Warning Cell
"""

    # HTML recommendations block
    rec_html_block = ""
    if recommendations:
        rec_li_items = "".join(f"<li style='margin-bottom: 6px;'>{r}</li>" for r in recommendations)
        rec_html_block = f"""
          <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 12px; margin: 16px 0; border-radius: 4px;">
            <p style="margin: 0 0 6px 0; font-weight: bold; color: #1e40af;">Academic Recommendations for Improvement:</p>
            <ul style="margin: 0; padding-left: 20px; color: #374151; font-size: 0.95em;">
              {rec_li_items}
            </ul>
          </div>
        """

    # Clean, responsive HTML format
    body_html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #1f2937; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 24px;">
          <h2 style="color: #1e3a8a; margin-top: 0; border-bottom: 2px solid {accent_color}; padding-bottom: 8px;">
            {header_title}
          </h2>
          <p>Dear <strong>{parent_salutation}</strong>,</p>
          <p>{opening_statement}</p>
          
          <table style="width: 100%; border-collapse: collapse; margin: 16px 0; background-color: #ffffff; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <tr style="background-color: #f1f5f9;">
              <td style="padding: 10px 14px; font-weight: bold; width: 40%;">Student Name</td>
              <td style="padding: 10px 14px;">{student_display}</td>
            </tr>
            <tr>
              <td style="padding: 10px 14px; font-weight: bold;">Predicted Performance</td>
              <td style="padding: 10px 14px; font-weight: bold; color: {accent_color}; font-size: 1.1em;">{predicted_score:.1f}%</td>
            </tr>
            <tr style="background-color: #f1f5f9;">
              <td style="padding: 10px 14px; font-weight: bold;">Academic Risk Level</td>
              <td style="padding: 10px 14px;"><span style="background-color: {accent_color}; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;">{alert_tier_badge}</span></td>
            </tr>
            <tr>
              <td style="padding: 10px 14px; font-weight: bold;">Current Attendance</td>
              <td style="padding: 10px 14px;">{attendance:.1f}%</td>
            </tr>
            <tr style="background-color: #f1f5f9;">
              <td style="padding: 10px 14px; font-weight: bold;">Study Hours Per Day</td>
              <td style="padding: 10px 14px;">{study_hours:.1f} hrs/day</td>
            </tr>
          </table>

          {rec_html_block}

          <div style="background-color: {banner_bg}; border-left: 4px solid {banner_border}; padding: 12px; margin: 16px 0; border-radius: 4px;">
            <p style="margin: 0; font-weight: bold; color: {banner_text_color};">Recommended Action for Parent/Guardian:</p>
            <p style="margin: 6px 0 0 0; color: #374151;">{parent_rec}</p>
          </div>

          <p>{core_message}</p>
          <p>{closing_statement}</p>

          <p style="margin-top: 24px; border-top: 1px solid #e2e8f0; padding-top: 12px; color: #64748b; font-size: 0.9em;">
            Regards,<br>
            <strong>Student Performance Prediction System</strong><br>
            Academic Early Warning Cell
          </p>
        </div>
      </body>
    </html>
    """

    return {
        "subject": subject,
        "body_text": body_text,
        "body_html": body_html,
        "parent_recommendation": parent_rec,
        "recommendations": recommendations or [],
    }



def get_smtp_credentials(custom_creds: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Safely retrieve SMTP credentials from custom parameter, Streamlit secrets, or OS environment variables.
    Reads: EMAIL_SENDER, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT.
    Does NOT hard-code any credentials. Never logs or displays passwords.
    Returns None if credentials are not configured.
    """
    if custom_creds and isinstance(custom_creds, dict):
        if custom_creds.get("username") and custom_creds.get("password"):
            return custom_creds

    # 1. Try streamlit secrets if available
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            # Check user-specified keys in st.secrets
            s_sender = st.secrets.get("EMAIL_SENDER") or (st.secrets.get("smtp", {}).get("sender") if "smtp" in st.secrets else None)
            s_user = st.secrets.get("EMAIL_SENDER") or (st.secrets.get("smtp", {}).get("username") if "smtp" in st.secrets else None)
            s_pass = st.secrets.get("EMAIL_PASSWORD") or (st.secrets.get("smtp", {}).get("password") if "smtp" in st.secrets else None)
            s_server = st.secrets.get("SMTP_SERVER") or (st.secrets.get("smtp", {}).get("server") if "smtp" in st.secrets else "smtp.gmail.com")
            s_port = st.secrets.get("SMTP_PORT") or (st.secrets.get("smtp", {}).get("port") if "smtp" in st.secrets else 587)

            if s_user and s_pass:
                return {
                    "server": str(s_server).strip(),
                    "port": int(s_port),
                    "username": str(s_user).strip(),
                    "password": str(s_pass).strip(),
                    "sender": str(s_sender or s_user).strip(),
                }
    except Exception:
        pass

    # 2. Try environment variables
    env_user = os.getenv("EMAIL_SENDER") or os.getenv("SMTP_USERNAME")
    env_pass = os.getenv("EMAIL_PASSWORD") or os.getenv("SMTP_PASSWORD")
    if env_user and env_pass:
        return {
            "server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "port": int(os.getenv("SMTP_PORT", 587)),
            "username": env_user,
            "password": env_pass,
            "sender": os.getenv("EMAIL_SENDER", env_user),
        }

    return None


def is_demo_mode(custom_creds: Optional[Dict[str, Any]] = None) -> bool:
    """
    Returns True if real email credentials are not configured, indicating Demo Mode.
    """
    return get_smtp_credentials(custom_creds) is None


def send_parent_alert_email(
    recipient_email: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
    smtp_credentials: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Send parent alert email using Gmail SMTP delivery on Port 587 with STARTTLS.
    If SMTP credentials are configured, sends real email via smtplib.
    If not configured, operates in safe Demo Mode and returns simulation status.

    Returns:
        dict: {
            "success": bool,
            "status": "sent" | "demo" | "error",
            "message": str,
            "recipient": str
        }
    """
    if not recipient_email or "@" not in recipient_email:
        return {
            "success": False,
            "status": "error",
            "message": "Invalid recipient email address provided.",
            "recipient": recipient_email,
        }

    creds = get_smtp_credentials(smtp_credentials)

    # DEMO MODE fallback
    if creds is None:
        return {
            "success": True,
            "status": "demo",
            "message": (
                f"Demo Mode: Parent alert generated successfully for {recipient_email}. "
                "(Simulated send — no real email dispatched because SMTP credentials are not configured in .streamlit/secrets.toml)."
            ),
            "recipient": recipient_email,
        }

    # REAL EMAIL DISPATCH VIA GMAIL SMTP (PORT 587 WITH STARTTLS)
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = creds["sender"]
        msg["To"] = recipient_email

        # Attach plain text and HTML versions
        msg.attach(MIMEText(body_text, "plain", "utf-8"))
        if body_html:
            msg.attach(MIMEText(body_html, "html", "utf-8"))

        server = str(creds.get("server", "smtp.gmail.com")).strip()
        port = int(creds.get("port", 587))

        # Connect using port 587 with STARTTLS for secure Gmail SMTP delivery
               if port == 465:
                    with smtplib.SMTP_SSL(
                        server, port, timeout=30
            ) as smtp:
                smtp.login(creds["username"], creds["password"])
                smtp.send_message(msg)
                else:
                     with smtplib.SMTP(
                     server, port, timeout=30
            ) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(creds["username"], creds["password"])
                smtp.send_message(msg)
        return {
            "success": True,
            "status": "sent",
            "message": f"Parent alert email dispatched successfully to {recipient_email}.",
            "recipient": recipient_email,
        }

    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "status": "error",
            "message": "SMTP Authentication failed. Please verify your email sender and app-specific password in .streamlit/secrets.toml.",
            "recipient": recipient_email,
        }
    except Exception as exc:
        return {
            "success": False,
            "status": "error",
            "message": f"Failed to send email via SMTP: {str(exc)}",
            "recipient": recipient_email,
        }
