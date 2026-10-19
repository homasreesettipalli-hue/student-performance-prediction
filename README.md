# AI-Based Student Performance Prediction and Early Warning System Using Machine Learning

An end-to-end Machine Learning web application designed to forecast student academic performance, assess risk tiers, detect individual weak areas, and generate dynamic, personalized academic recommendations before final examinations.

---

## 1. Project Title & Overview
**Title:** AI-Based Student Performance Prediction and Early Warning System Using Machine Learning  
**Application:** Student Performance Predictor (Streamlit Web Dashboard)  
**Target Audience:** Students, Faculty Mentors, and Academic Advisors  
**Academic Level:** Suitable for B.Tech Minor/Major Project, Capstone, or Learning Analytics research.

---

## 2. Project Objective
Traditional academic grading is post-hoc: it reveals poor outcomes only after examinations conclude. This project builds a proactive early warning system that combines:
1. **Historical Academic Indicators** (Previous GPA, Internal Exam marks, Quiz scores, Assignment grades)
2. **Continuous Learning Behavior Indicators** (Attendance, Study hours per day, Assignment completion rate, Quiz participation, LMS activity)

The system computes a normalized **Learning Behavior Index (LBI)**, predicts the student's expected final score percentage via regression machine learning models, classifies risk (Low, Medium, High), isolates weak learning factors, provides targeted student recommendations, and triggers a dedicated **Parent Alert System** for high-risk students to enable timely home-school intervention.

### System Workflow
```text
Student Academic & Behavioral Data
       ↓
Data Preprocessing (Median Imputation & Standardization)
       ↓
Machine Learning Prediction (Regression Benchmark)
       ↓
Predicted Final Performance & Learning Behavior Index (LBI)
       ↓
Academic Risk Classification (Low / Medium / High)
       ↓
Early Warning & Factor-Specific Recommendations
       ↓
Parent Alert System (Triggered for High-Risk Cases)
```


---

## 3. Problem Statement
Educational institutions often struggle to identify students at academic risk mid-semester when corrective interventions are still possible. Existing assessments are siloed and fail to synthesize behavioral engagement with prior test history. This system addresses this gap with an intuitive machine-learning-driven early warning dashboard.

---

## 4. Input Features & Validation

### Academic Factors
| Feature | Range / Scale | Description |
|---|---|---|
| `Previous GPA / CGPA` | `0.0 – 10.0` | Cumulative grade point average from past semesters |
| `Internal Examination Marks` | `0.0 – 100.0%` | Mid-term internal assessment marks |
| `Previous Subject Marks` | `0.0 – 100.0%` | Prior score in prerequisite/related coursework |
| `Quiz Marks` | `0.0 – 100.0%` | Formative evaluation quiz average |
| `Assignment Marks` | `0.0 – 100.0%` | Quality and correctness score on submitted assignments |

### Learning Behavior Factors
| Feature | Range / Scale | Description |
|---|---|---|
| `Attendance Percentage` | `0.0 – 100.0%` | Lecture and lab attendance rate |
| `Study Hours Per Day` | `0.0 – 14.0 hrs` | Self-directed study hours outside scheduled classes |
| `Assignment Completion Rate` | `0.0 – 100.0%` | Timely submission compliance rate |
| `Quiz Participation` | `0.0 – 100.0%` | Participation frequency in scheduled quizzes |
| `Learning / LMS Activity` | `0.0 – 100.0` | Engagement score on LMS resources & course modules |

---

## 5. Learning Behavior Index (LBI) Formulation
The composite **Learning Behavior Index (LBI)** combines continuous effort indicators into a standardized 0–100 scale:

$$\text{LBI} = 0.30 \times \text{Quiz Performance} + 0.25 \times \text{Assignment Completion} + 0.20 \times \text{Attendance} + 0.15 \times \text{Study Hours}_{\text{norm}} + 0.10 \times \text{Learning Activity}_{\text{norm}}$$

- **Study Hours Normalization:** $\min\left(\frac{\text{Study Hours}}{10.0} \times 100, 100\right)$
- **Learning Activity Normalization:** Bounded within $[0, 100]$.
- **Quiz Performance:** Formative performance combining quiz marks and quiz participation.

---

## 6. Academic Risk Classification & Notification Policy
Predicted final percentage scores are categorized into actionable intervention tiers:

| Predicted Final Score | Academic Risk Level | Student & Faculty Advisory | Parent Alert Policy |
|---|---|---|---|
| **75% – 100%** | 🟢 **LOW RISK** | Good standing; maintain study consistency and explore peer tutoring. | **No alert sent** (Student performing satisfactorily). |
| **60% – 74%** | 🟡 **MEDIUM RISK** | Advisory warning; address targeted weak areas (quizzes, assignments, revision). | **Early-Warning Parent Alert** (Polite notice that performance requires attention). |
| **Below 60%** | 🔴 **HIGH RISK** | Critical early warning; faculty mentor consultation required immediately. | **Urgent Parent Alert** (Immediate academic intervention & counseling recommended). |

---

## 7. Machine Learning Algorithms & Evaluation

Four regression models are trained and benchmarked:
1. **Linear Regression** (Interpretable baseline)
2. **Decision Tree Regressor** (Captures nonlinear threshold splits)
3. **Random Forest Regressor** (Ensemble of decision trees reducing variance)
4. **Support Vector Regression (SVR)** (Radial basis function kernel for complex boundaries)

### Evaluation Metrics
- **MAE (Mean Absolute Error):** Measures average absolute deviation between actual and predicted scores.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction discrepancies.
- **$R^2$ Score (Coefficient of Determination):** Proportion of variance explained by model inputs.

### Benchmark Results (Independent Test Split $N=300$)
| Model | MAE | RMSE | $R^2$ Score | Selection |
|---|---|---|---|---|
| **Linear Regression** | **1.8261** | **2.2580** | **0.9772** | 🏆 **Best Model** |
| Support Vector Regression (SVR) | 2.0074 | 2.4903 | 0.9723 | Candidate |
| Random Forest Regressor | 2.1671 | 2.7735 | 0.9657 | Candidate |
| Decision Tree Regressor | 2.6256 | 3.3832 | 0.9489 | Candidate |

---

## 8. Weak Area Identification & Rule-Based Recommendations

The system audits each student parameter against benchmark criteria:
- **Attendance < 75%** $\rightarrow$ *"Improve attendance and maintain regular participation in classes (target at least 75%–85%)."*
- **Study Hours < 2.5 hrs/day** $\rightarrow$ *"Increase daily study time and follow a structured study schedule (aim for 3 to 4 hours of focused study daily)."*
- **Quiz Marks < 60%** $\rightarrow$ *"Practice additional quizzes and revise difficult topics prior to upcoming assessments."*
- **Assignment Completion < 70%** $\rightarrow$ *"Complete pending assignments and maintain timely submission to secure full internal assessment credit."*
- **LMS Activity < 50/100** $\rightarrow$ *"Increase participation in learning activities and regularly use available LMS learning resources and video lectures."*
- **Internal Marks < 50%** $\rightarrow$ *"Review past internal exam mistakes, clarify doubts with course instructors during office hours, and focus on core technical concepts."*

---

## 9. Automatic Parent Alert System

The **Parent Alert System** connects campus performance analytics with at-home parental guidance. The system provides an **automated, multi-tiered parent notification mechanism** that triggers immediately upon completing a prediction:

1. **🟢 Low Risk ($\ge 75\%$): No Parent Alert**
   - Displays that the student is performing satisfactorily.
   - Routine semester tracking continues without sending any parent notification.

2. **🟡 Medium Risk ($60\% - 74\%$): Automatic Early-Warning Parent Alert**
   - Automatically dispatched upon prediction completion (no manual button click required).
   - Generates a polite, supportive early-warning advisory informing parents that performance requires attention before final exams.
   - Includes student name, predicted score, risk level, attendance, daily study hours, personalized academic recommendations, and constructive parent guidance.
   - **Does not treat Medium Risk as an emergency.**

3. **🔴 High Risk ($< 60\%$): Automatic Urgent Parent Alert**
   - Automatically dispatched upon prediction completion (no manual button click required).
   - Generates a strong, prioritized urgent alert indicating that immediate academic counseling and faculty mentor consultation are recommended.
   - Includes student name, predicted score, risk level, attendance, daily study hours, personalized academic recommendations, and specific corrective actions.

### Email Notification Formats

#### A. Medium Risk Notification (Polite Advisory)
```text
Subject: Student Performance Early-Warning Advisory - [Student Name]

Dear [Parent/Guardian Name],

This is a polite early-warning academic progress update regarding your ward's performance.

Student Name: [Student Name]
Predicted Performance: [Predicted Score]%
Risk Level: MEDIUM RISK
Attendance: [Attendance]%
Daily Study Hours: [Study Hours] hrs/day

Academic Recommendations for Student:
  • [Personalized recommendation item 1]
  • [Personalized recommendation item 2]

The system has identified that the student's performance requires attention to ensure steady progress toward examination milestones.

Recommended Action for Parent/Guardian:
[Constructive study and revision support guidance]

Please encourage regular study routines and discuss progress with your ward to help them improve before final examinations.

Regards,
Student Performance Prediction System
Academic Early Warning Cell
```

#### B. High Risk Notification (Urgent Academic Alert)
```text
Subject: URGENT: Student Performance Academic Alert - [Student Name]

Dear [Parent/Guardian Name],

This is an urgent early warning regarding your ward's projected academic performance.

Student Name: [Student Name]
Predicted Performance: [Predicted Score]%
Risk Level: HIGH RISK
Attendance: [Attendance]%
Daily Study Hours: [Study Hours] hrs/day

Academic Recommendations for Student:
  • ⚠️ URGENT INTERVENTION: Schedule an academic counseling session with your faculty mentor or advisor immediately.
  • [Personalized recommendation item 1]
  • [Personalized recommendation item 2]

The system has identified critical academic risks that require immediate academic attention and faculty counseling.

Recommended Action for Parent/Guardian:
[Urgent counseling and faculty advisor consultation guidance]

Please treat this notification with priority, monitor attendance and study routines closely, and schedule a consultation with the academic mentor.

Regards,
Student Performance Prediction System
Academic Early Warning Cell
```

### Streamlit Rerun & Duplicate Send Prevention
Streamlit applications rerun the script frequently whenever widgets are interacted with. To guarantee reliable behavior:
- **Event-Based Dispatch Tracking:** Each explicit submission via `"🔮 Predict Performance"` creates a unique `prediction_event_id` stored in `st.session_state`.
- **Zero Duplicate Sends on Reruns:** Dispatches are logged in `st.session_state["email_dispatched_events"]`. Ordinary reruns (tab switches, sidebar changes, widget interactions) detect that the event was already handled and **never dispatch repeated emails**.
- **New Prediction Events:** If the user submits new student details, a fresh event ID is generated and processed once.
- **Transparent Failure & Safe Retry:** If SMTP transmission encounters network or authentication issues, an explicit error banner is displayed with a **"🔄 Retry Sending Parent Alert"** button allowing safe retry. Success is never falsely reported if SMTP fails.

### Safe Demo Mode vs Live SMTP Delivery
- **Safe Demo Mode (Default Fallback):** If live SMTP credentials are not configured in `.streamlit/secrets.toml`, the system operates in an offline demonstration mode. It validates inputs, formats complete templates, and provides clear simulation status feedback without outbound network errors.
- **Live SMTP Delivery:** When email credentials are provided, alerts are automatically dispatched via Gmail SMTP on Port 587 using STARTTLS.

### 4. Configuring Email Credentials Safely
Never hard-code passwords or tokens into the codebase. Keep `.streamlit/secrets.toml` listed in `.gitignore` to prevent committing secrets to version control.

**Option A — Streamlit Secrets (`.streamlit/secrets.toml`):**
```toml
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_16_character_app_password"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
```

**Option B — Environment Variables:**
```bash
set EMAIL_SENDER=your_email@gmail.com
set EMAIL_PASSWORD=your_16_character_app_password
set SMTP_SERVER=smtp.gmail.com
set SMTP_PORT=587
```

---

## 10. Project Directory Structure
```text
student-performance-prediction/
│
├── app.py                     # Streamlit interactive web application with Parent Alert UI
├── run_app.bat                # Windows 1-click launcher script
├── train_model.py             # Script to train, benchmark, and save regression models
├── generate_dataset.py        # Logically consistent synthetic dataset generator
├── student_performance.csv    # Generated 1,500 record dataset
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive project documentation
│
├── models/
│   ├── best_model.pkl         # Serialized production model (Linear Regression)
│   ├── preprocessor.pkl       # Serialized StandardScaler & SimpleImputer
│   ├── model_metrics.json     # Saved comparison metrics and benchmark data
│   ├── linear_regression.pkl  # Candidate model artifact
│   ├── decision_tree.pkl      # Candidate model artifact
│   ├── random_forest.pkl      # Candidate model artifact
│   └── support_vector_regression_svr.pkl
│
├── src/
│   ├── __init__.py            # Package initializer & exports
│   ├── parent_alert.py        # Parent Alert System (template generator & safe dispatcher)
│   ├── preprocessing.py       # Data cleaning, scaling, and splitting pipeline
│   ├── feature_engineering.py # LBI calculation and normalization routines
│   ├── recommendations.py     # Rule-based weak area and advisory engine
│   └── prediction.py          # Unified inference pipeline
│
└── notebooks/
    └── model_analysis.ipynb   # Complete exploratory data analysis & visualization notebook
```

---

## 10. Installation & Setup

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, 3.13, or 3.14 installed on your system.

### Step 1: Clone or Navigate to the Directory
```bash
cd student-performance-prediction
```

### Step 2: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

---

## 11. How to Generate Dataset & Train Models

### Generate the Dataset
To regenerate the 1,500 logically consistent student records:
```bash
python generate_dataset.py
```

### Train and Evaluate All 4 Models
```bash
python train_model.py
```
This trains all four models, prints the comparison table, selects the best model, and saves all artifacts to `models/`.

---

## 12. How to Run the Streamlit Application

Launch the Streamlit web dashboard:
```bash
python -m streamlit run app.py
```
Or if `streamlit` is in your system PATH:
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 13. Example Input & Output

### Example Input
- **Previous GPA:** `8.0` / 10.0
- **Internal Examination Marks:** `72.0%`
- **Previous Subject Marks:** `70.0%`
- **Quiz Marks:** `75.0%`
- **Assignment Marks:** `85.0%`
- **Attendance Percentage:** `78.0%`
- **Study Hours/Day:** `3.0 hrs`
- **Assignment Completion Rate:** `85.0%`
- **Quiz Participation:** `80.0%`
- **Learning/LMS Activity:** `65.0`

### Expected Output
```text
=====================================================
Predicted Final Score   : 74.1%
Learning Behavior Index : 70.8
Risk Level              : MEDIUM RISK
-----------------------------------------------------
Identified Weak Areas   : None (Above critical thresholds)
Personalized Advice     : Good progress, but incremental consistency in 
                          daily revision will push performance into Low Risk tier (>=75%).
=====================================================
```

---

## 14. Verification & Testing Scenarios

You can verify the entire pipeline using the **⚡ Quick Preset Profiles** in the sidebar:

### Test 1 — Low Risk Case
- **Profile:** Select **High Achiever (Low Risk)** (or set GPA $\ge 8.5$, Internal $\ge 85\%$, Attendance $\ge 90\%$).
- **Click:** `"🔮 Predict Performance"`
- **Expected Outcome:**
  - Predicted Final Score $\ge 75\%$
  - Risk Level: `LOW RISK` (🟢)
  - Existing recommendations display positive reinforcement.
  - **Parent Alert Status:** Displays *"Satisfactory Performance: Student is currently performing satisfactorily at low academic risk. No parent alert email required or sent."*
  - **Email Behavior:** No email is dispatched (neither live nor demo).

### Test 2 — Medium Risk Case (Automatic Early-Warning Alert)
- **Profile:** Select **Average Student (Medium Risk)** (or set GPA $\approx 7.2$, Internal $\approx 68\%$, Attendance $\approx 78\%$).
- **Click:** `"🔮 Predict Performance"`
- **Expected Outcome:**
  - Predicted Final Score between $60\%$ and $74\%$
  - Risk Level: `MEDIUM RISK` (🟡)
  - Existing recommendations provide targeted study advice.
  - **Automatic Email Dispatch:** Email is dispatched **automatically upon prediction** (no manual "Send" button needed).
  - Displays a clean, unobtrusive success indicator: `✓ Parent alert sent successfully`.
  - Displays a compact risk badge (`[MEDIUM RISK]`) right next to the **Parent Alert Summary** title.
  - Populates Student & Parent details, displays academic indicators (attendance, daily study hours), and formats the polite advisory email preview including student academic recommendations.

### Test 3 — High Risk Case (Automatic Urgent Academic Alert)
- **Profile:** Select **At-Risk Student (High Risk)** (or set GPA $\le 5.0$, Internal $\le 45\%$, Attendance $\le 55\%$).
- **Click:** `"🔮 Predict Performance"`
- **Expected Outcome:**
  - Predicted Final Score $< 60\%$
  - Risk Level: `HIGH RISK` (🔴)
  - Existing recommendations flag urgent faculty mentor intervention.
  - **Automatic Email Dispatch:** Email is dispatched **automatically upon prediction** (no manual "Send" button needed).
  - Displays a clean, unobtrusive success indicator: `✓ Parent alert sent successfully`.
  - Displays a compact risk badge (`[HIGH RISK]`) right next to the **Parent Alert Summary** title.
  - Populates Student & Parent details, displays academic indicators (attendance, daily study hours), and formats the urgent academic alert email preview including student academic recommendations.

### Test 4 — Streamlit Reruns (Duplicate Send Prevention)
- After generating a Medium Risk or High Risk prediction, interact with unrelated UI widgets (e.g. switch between tabs, toggle expanders, or change sidebar controls without pressing predict).
- **Expected Outcome:**
  - The application reruns normally.
  - **No duplicate emails are sent.** The session state tracks the `current_prediction_event_id` and bypasses repeated dispatch.
  - If `"🔮 Predict Performance"` is explicitly clicked again, it generates a new prediction event and dispatches once for the new run.

### Test 5 — Automated Test Suite Execution
Run the comprehensive test script to verify all 6 requirements programmatically:
```bash
py test_auto_parent_alert.py
```
This suite verifies:
1. Low Risk student prediction & email exclusion.
2. Medium Risk early-warning alert content & recommendations.
3. High Risk urgent alert content & recommendations.
4. Streamlit rerun simulation & duplicate send prevention.
5. SMTP failure handling and error responses.
6. Demo mode fallback operation.

