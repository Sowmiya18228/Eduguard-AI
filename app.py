import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
import tempfile
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="EduGuard AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .stApp { background: #0b0f1a; color: #e8eaf0; }
    section[data-testid="stSidebar"] { background: #0d1120; border-right: 1px solid #1e2540; }
    section[data-testid="stSidebar"] * { color: #c9cde0 !important; }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="metric-container"] {
        background: #111827; border: 1px solid #1e2d4a;
        border-radius: 12px; padding: 1rem;
    }
    [data-testid="metric-container"] label {
        color: #8891aa !important; font-size: 12px !important;
        letter-spacing: 0.08em; text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e8eaf0 !important; font-size: 26px !important; font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #111827; border-radius: 10px;
        padding: 4px; gap: 4px; border: 1px solid #1e2540;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 8px; color: #8891aa;
        font-weight: 500; font-size: 13px; padding: 8px 16px; border: none;
    }
    .stTabs [aria-selected="true"] { background: #1a2540 !important; color: #4f9cff !important; }
    .stNumberInput input, .stTextInput input {
        background: #111827 !important; border: 1px solid #1e2540 !important;
        border-radius: 8px !important; color: #e8eaf0 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white;
        border: none; border-radius: 10px; font-weight: 600; font-size: 14px;
        padding: 0.65rem 2rem; width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        transform: translateY(-1px); box-shadow: 0 8px 24px rgba(37,99,235,0.3);
    }
    .stDownloadButton > button {
        background: #111827; color: #4f9cff; border: 1px solid #1e2d4a;
        border-radius: 10px; font-weight: 600; width: 100%;
    }
    .stProgress > div > div { background: linear-gradient(90deg,#1d4ed8,#4f9cff); border-radius:99px; }
    .stProgress > div { background:#1e2540; border-radius:99px; height:10px !important; }
    .stSelectbox > div > div {
        background: #111827 !important; border: 1px solid #1e2540 !important;
        border-radius: 8px !important; color: #e8eaf0 !important;
    }
    .section-card {
        background: #111827; border: 1px solid #1e2540;
        border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem;
    }
    .risk-high {
        background: rgba(220,38,38,0.12); border: 1px solid rgba(220,38,38,0.35);
        color: #f87171; border-radius: 99px; padding: 6px 18px;
        font-size: 13px; font-weight: 600; display: inline-block; letter-spacing: 0.06em;
    }
    .risk-safe {
        background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.35);
        color: #34d399; border-radius: 99px; padding: 6px 18px;
        font-size: 13px; font-weight: 600; display: inline-block; letter-spacing: 0.06em;
    }
    .field-label {
        font-size: 12px; color: #8891aa; font-weight: 500;
        letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px;
    }
    hr { border-color: #1e2540 !important; margin: 1.5rem 0; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0b0f1a; }
    ::-webkit-scrollbar-thumb { background: #1e2540; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# PLOTLY THEME
# ---------------------------------------------------

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0f172a",
    font=dict(family="Sora, sans-serif", color="#c9cde0", size=12),
    title_font=dict(family="Sora, sans-serif", size=15, color="#e8eaf0"),
    colorway=["#4f9cff","#34d399","#fb923c","#a78bfa","#f472b6"],
    xaxis=dict(gridcolor="#1e2540", linecolor="#1e2540", zeroline=False),
    yaxis=dict(gridcolor="#1e2540", linecolor="#1e2540", zeroline=False),
    margin=dict(l=16, r=16, t=40, b=16),
)

def pt():
    """PLOT_THEME without margin — avoids duplicate keyword error"""
    return {k: v for k, v in PLOT_THEME.items() if k != "margin"}

# ---------------------------------------------------
# LOAD MODEL + DATA
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/dropout_model.pkl")

@st.cache_data
def load_cleaned():
    return pd.read_csv("data/processed/cleaned_student_data.csv")

@st.cache_data
def load_featured():
    return pd.read_csv("data/processed/featured_student_data.csv")

@st.cache_data
def load_raw():
    return pd.read_csv("data/raw/student_data.csv", sep=";")

model   = load_model()
df      = load_cleaned()      # 34 cols — primary
df_feat = load_featured()     # 37 cols — has performance_score, attendance_score
df_raw  = load_raw()          # original semicolon-separated raw data

# ---------------------------------------------------
# CONSTANTS
# ---------------------------------------------------

FEATURE_COLS  = ["studytime", "failures", "absences", "G1", "G2"]
FEATURE_NAMES = ["Study Time", "Failures", "Absences", "Exam 1", "Exam 2"]
TARGET_COL    = "dropout_risk"

total_students = df.shape[0]
high_risk      = int(df[TARGET_COL].sum())
safe_students  = total_students - high_risk
risk_pct_side  = round(high_risk / total_students * 100, 1)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.markdown("## 🎓 EduGuard AI")
    st.caption("Student Dropout Risk Intelligence Platform")

    st.markdown("---")

    st.markdown("### 📋 System Info")

    st.write("**Model:** Random Forest")
    st.write("**Input Parameters:** 5")
    st.write("**Prediction:** Active")
    st.write("**Bulk Analysis:** Enabled")

    st.markdown("---")

    st.markdown("### 🚀 Available Modules")

    st.success("⚡ Individual Prediction")
    st.success("💬 Student Support")
    st.success("📂 Bulk Prediction")

    st.markdown("---")

    st.info(
        """
        EduGuard AI helps identify
        students at risk of dropout
        using machine learning
        predictions and intervention
        recommendations.
        """
    )

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

col_logo, col_title = st.columns([1, 10])
with col_logo:
    if os.path.exists("app/assets/logo.jpeg"):
        st.image(Image.open("app/assets/logo.jpeg"), width=56)
with col_title:
    st.markdown("""
        <h1 style='font-size:26px;font-weight:700;margin:0;color:#e8eaf0;'>
        EduGuard <span style='color:#4f9cff;'>AI</span></h1>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "⚡ Prediction",
    "💬 Student Support",
    "📁 Bulk Prediction"
])

# ===================================================
# TAB 1 — PREDICTION
# ===================================================

with tab1:

    st.markdown(
        "<p style='color:#8891aa;font-size:14px;margin-bottom:20px;'>"
        "Enter student academic details to generate an AI-powered dropout risk assessment."
        "</p>",
        unsafe_allow_html=True
    )

    col_form, col_result = st.columns([1.1, 1], gap="large")

    # -----------------------------
    # INPUT FORM
    # -----------------------------
    with col_form:

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)

        st.markdown(
            "<p style='font-size:13px;font-weight:600;color:#e8eaf0;"
            "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:16px;'>"
            "Student Profile</p>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p class='field-label'>Study Time (1=Low, 4=High)</p>",
            unsafe_allow_html=True
        )

        studytime = st.slider(
            "Study Time",
            1,
            4,
            2,
            label_visibility="collapsed"
        )

        st.markdown(
            "<p class='field-label'>Past Academic Failures</p>",
            unsafe_allow_html=True
        )

        failures = st.slider(
            "Failures",
            0,
            3,
            0,
            label_visibility="collapsed"
        )

        st.markdown(
            "<p class='field-label'>Total Absences (0–15)</p>",
            unsafe_allow_html=True
        )

        absences = st.number_input(
            "Absences",
            0,
            15,
            0,
            label_visibility="collapsed"
        )

        cg1, cg2 = st.columns(2)

        with cg1:

            st.markdown(
                "<p class='field-label'>Internal Exam 1 (0–20)</p>",
                unsafe_allow_html=True
            )

            G1 = st.number_input(
                "G1",
                0,
                20,
                10,
                label_visibility="collapsed"
            )

        with cg2:

            st.markdown(
                "<p class='field-label'>Internal Exam 2 (0–20)</p>",
                unsafe_allow_html=True
            )

            G2 = st.number_input(
                "G2",
                0,
                20,
                10,
                label_visibility="collapsed"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        predict_btn = st.button(
            "⚡ Run Risk Analysis",
            use_container_width=True
        )

        if predict_btn:

            inp_arr = np.array([
                [studytime, failures, absences, G1, G2]
            ])

            prediction = model.predict(inp_arr)[0]

            risk_prob = model.predict_proba(inp_arr)[0][1]

            st.session_state.prediction = prediction
            st.session_state.risk_prob = risk_prob

            st.session_state.inputs = {
                "studytime": studytime,
                "failures": failures,
                "absences": absences,
                "G1": G1,
                "G2": G2
            }

        # =============================
    # RESULT PANEL
    # =============================

    with col_result:

        if "prediction" in st.session_state:

            pred = st.session_state.prediction
            risk_prob = st.session_state.risk_prob
            inp = st.session_state.inputs

            rpv = round(risk_prob * 100, 1)

            sc = (
                "#f87171"
                if rpv > 60
                else "#fb923c"
                if rpv > 35
                else "#34d399"
            )

            st.markdown(
                "<div class='section-card'>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<p style='font-size:13px;font-weight:600;color:#e8eaf0;"
                "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:16px;'>"
                "Risk Assessment</p>",
                unsafe_allow_html=True
            )

            if pred == 1:
                st.markdown(
                    "<span class='risk-high'>⚠ HIGH DROPOUT RISK</span>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<span class='risk-safe'>✓ LOW RISK — SAFE</span>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
                <div style='margin:20px 0 8px;'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                        <span style='font-size:13px;color:#8891aa;'>Risk Score</span>
                        <span style='font-size:20px;font-weight:700;color:{sc};'>
                            {rpv}%
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(int(risk_prob * 100))

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            # -------------------------
            # RECOMMENDATIONS
            # -------------------------

            st.markdown(
                "<div class='section-card'>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<p style='font-size:13px;font-weight:600;color:#e8eaf0;"
                "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px;'>"
                "Recommended Actions</p>",
                unsafe_allow_html=True
            )

            recs = []

            if inp["absences"] > 10:
                recs.append(
                    ("📅", "Critical: Very low attendance — immediate follow-up needed")
                )
            elif inp["absences"] > 5:
                recs.append(
                    ("📅", "Improve class attendance regularly")
                )

            if inp["failures"] > 1:
                recs.append(
                    ("📚", "Arrange targeted academic tutoring")
                )
            elif inp["failures"] == 1:
                recs.append(
                    ("📚", "Review weak subjects with teacher support")
                )

            if inp["studytime"] < 2:
                recs.append(
                    ("⏱", "Build a structured daily study routine")
                )

            if inp["G1"] < 10 or inp["G2"] < 10:
                recs.append(
                    ("📝", "Focus on internal exam preparation")
                )

            if risk_prob < 0.35:
                recs.append(
                    ("✅", "Maintain current academic performance")
                )

            if risk_prob > 0.70:
                recs.append(
                    ("🚨", "Immediate counsellor intervention needed")
                )

            for icon, rec in recs:
                st.write(f"{icon} {rec}")

            



# ===================================================
# TAB 2 — STUDENT SUPPORT CHATBOT
# ===================================================

with tab2:

    st.markdown(
        """
        <p style='color:#8891aa;font-size:14px;margin-bottom:20px;'>
        AI-powered student counselor using Groq Llama 3.
        </p>
        """,
        unsafe_allow_html=True
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input(
        "Ask about studies, attendance, exams, stress, motivation..."
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        prediction_context = ""

        if "risk_prob" in st.session_state:

            inp = st.session_state.inputs

            prediction_context = f"""
            Student Risk Information

            Risk Probability:
            {st.session_state.risk_prob*100:.2f}%

            Study Time:
            {inp['studytime']}

            Failures:
            {inp['failures']}

            Absences:
            {inp['absences']}

            Internal Exam 1:
            {inp['G1']}

            Internal Exam 2:
            {inp['G2']}
            """

        with st.spinner("EduGuard AI is thinking..."):

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role":"system",
                        "content":"You are EduGuard AI."
                    },
                    {
                        "role":"user",
                        "content":question
                    }
                ]           
            )

            answer = response.choices[0].message.content

        st.session_state.chat_history.append(
            {
                "role":"assistant",
                "content":answer
            }
        )

        st.rerun()





# ===================================================
# TAB 3 — BULK STUDENT PREDICTION
# ===================================================

with tab3:

    st.header("📂 Bulk Student Risk Prediction")

    st.info(
        """
        📌 Required Columns:
        • failures
        • absences
        • G1 (Internal Mark 1)
        • G2 (Internal Mark 2)

        📌 Optional Columns:
        • reg_no
        • name
        • department
        • year

        📌 Supported Alternative Names:
        • Attendance → absences
        • Internal Mark 1 / IAT1 / CIA1 → G1
        • Internal Mark 2 / IAT2 / CIA2 → G2

        Supported file formats: CSV (.csv) and Excel (.xlsx)
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Student CSV / Excel File",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            # -------------------------------
            # READ FILE
            # -------------------------------

            if uploaded_file.name.endswith(".csv"):

                bulk_df = pd.read_csv(
                    uploaded_file,
                    sep=None,
                    engine="python"
                )

            else:

                bulk_df = pd.read_excel(uploaded_file)
                

            bulk_df.columns = bulk_df.columns.str.strip()

            # -------------------------------
            # AUTO COLUMN MAPPING
            # -------------------------------

            column_mapping = {

                # Register Number
                "Register No": "reg_no",
                "Reg No": "reg_no",
                "Roll No": "reg_no",
                "Student ID": "reg_no",

                # Name
                "Name": "name",
                "Student Name": "name",
                "Student_Name": "name",

                # Internal Marks
                "CIA1": "G1",
                "IAT1": "G1",
                "Internal1": "G1",
                "Internal Mark 1": "G1",

                "CIA2": "G2",
                "IAT2": "G2",
                "Internal2": "G2",
                "Internal Mark 2": "G2",

                # Failures
                "Arrears": "failures",
                "Backlogs": "failures",
                "Failed Subjects": "failures",

                # Attendance
                "Attendance": "attendance",
                "Attendance %": "attendance",
                "Attendance Percentage": "attendance"
            }

            bulk_df.rename(
                columns=column_mapping,
                inplace=True
            )

            st.success("✅ File Loaded Successfully")

            st.subheader("Uploaded Data")

            st.dataframe(
                bulk_df,
                use_container_width=True
            )

            # -------------------------------
            # PREDICT BUTTON
            # -------------------------------

            if st.button(
                "🚀 Predict All Students",
                use_container_width=True
            ):

                # ---------------------------
                # ATTENDANCE -> ABSENCES
                # ---------------------------

                if (
                    "attendance" in bulk_df.columns
                    and "absences" not in bulk_df.columns
                ):

                    bulk_df["absences"] = (
                        (100 - bulk_df["attendance"]) / 5
                    ).round()

                # ---------------------------
                # AUTO STUDYTIME
                # ---------------------------

                if "studytime" not in bulk_df.columns:

                    if "attendance" in bulk_df.columns:

                        bulk_df["studytime"] = np.where(
                            bulk_df["attendance"] >= 85,
                            4,
                            np.where(
                                bulk_df["attendance"] >= 70,
                                3,
                                2
                            )
                        )

                    else:

                        bulk_df["studytime"] = 2

                # ---------------------------
                # REQUIRED COLUMNS CHECK
                # ---------------------------

                required_cols = [
                    "studytime",
                    "failures",
                    "absences",
                    "G1",
                    "G2"
                ]

                missing = [
                    col
                    for col in required_cols
                    if col not in bulk_df.columns
                ]

                if missing:

                    st.error(
                        f"❌ Missing Required Columns: {missing}"
                    )

                    st.stop()

                # ---------------------------
                # MODEL INPUT
                # ---------------------------

                X = bulk_df[
                    [
                        "studytime",
                        "failures",
                        "absences",
                        "G1",
                        "G2"
                    ]
                ]

                # ---------------------------
                # PREDICTION
                # ---------------------------

                bulk_df["Risk"] = model.predict(X)

                bulk_df["Risk Probability"] = (
                    model.predict_proba(X)[:, 1] * 100
                ).round(2)

                bulk_df["Status"] = bulk_df["Risk"].map({
                    0: "SAFE",
                    1: "HIGH RISK"
                })

                # ---------------------------
                # SAVE FOR OTHER TABS
                # ---------------------------

                st.session_state["latest_results"] = bulk_df.copy()

                bulk_df.to_csv(
                   "prediction_results.csv",
                    index=False
                )

                st.success(
                   "✅ Prediction Completed Successfully"
                )
                

                # ---------------------------
                # DISPLAY RESULT
                # ---------------------------

                display_cols = []

                for col in [
                    "reg_no",
                    "name",
                    "department",
                    "year"
                ]:

                    if col in bulk_df.columns:
                        display_cols.append(col)

                display_cols += [
                    "Risk Probability",
                    "Status"
                ]

                st.subheader(
                    "📊 Student Risk Report"
                )

                st.dataframe(
                    bulk_df[display_cols],
                    use_container_width=True
                )

                # ---------------------------
                # HIGH RISK STUDENTS
                # ---------------------------

                high_risk = bulk_df[
                    bulk_df["Status"] == "HIGH RISK"
                ]

                st.subheader(
                    "🚨 High Risk Students"
                )

                if len(high_risk) > 0:

                    risk_cols = []

                    for col in [
                        "reg_no",
                        "name",
                        "department",
                        "year"
                    ]:

                        if col in high_risk.columns:
                            risk_cols.append(col)

                    risk_cols += [
                        "Risk Probability",
                        "Status"
                    ]

                    st.dataframe(
                        high_risk[risk_cols],
                        use_container_width=True
                    )

                else:

                    st.success(
                        "No High Risk Students Found"
                    )

                # ---------------------------
                # DOWNLOAD REPORT
                # ---------------------------

                csv_report = bulk_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇ Download Full Report",
                    data=csv_report,
                    file_name="student_risk_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Error Loading File: {e}"
            )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:12px;color:#4a5270;'>"
    "EduGuard AI &nbsp;·&nbsp; AI-Powered Student Dropout Prevention Platform"
    "</p>", unsafe_allow_html=True)