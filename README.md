import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq

# ------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="HemoScan AI – Anemia Detection",
    page_icon="🩸",
    layout="wide"
)

# ------------------------------------------------------------------------------
# Helper functions for anemia detection
# ------------------------------------------------------------------------------
def detect_anemia(hemoglobin, gender, age):
    """
    Detect anemia based on WHO hemoglobin thresholds.
    Returns a tuple: (is_anemic, severity)
    Severity: 'Normal', 'Mild', 'Moderate', 'Severe'
    """
    if gender == "Male":
        if hemoglobin < 13:
            if hemoglobin < 8:
                severity = "Severe"
            elif hemoglobin < 11:
                severity = "Moderate"
            else:
                severity = "Mild"
            return True, severity
        else:
            return False, "Normal"
    else:  # Female
        if hemoglobin < 12:
            if hemoglobin < 8:
                severity = "Severe"
            elif hemoglobin < 11:
                severity = "Moderate"
            else:
                severity = "Mild"
            return True, severity
        else:
            return False, "Normal"

def classify_morphology(mcv, mch, mchc):
    """
    Classify anemia morphology based on red cell indices.
    Returns a string describing the type.
    """
    if mcv < 80:
        size = "microcytic"
    elif mcv > 100:
        size = "macrocytic"
    else:
        size = "normocytic"

    if mchc < 32:
        chromia = "hypochromic"
    else:
        chromia = "normochromic"

    return f"{size} {chromia} anemia"

# ------------------------------------------------------------------------------
# Groq client and report generation
# ------------------------------------------------------------------------------
def generate_groq_report(patient_data, diagnosis, morphology):
    """
    Use Groq LLM to generate a patient-friendly report with risk analysis and recommendations.
    """
    # Retrieve API key from secrets
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("Groq API key not found. Please set it in Streamlit secrets.")
        return None

    client = Groq(api_key=api_key)

    # Construct prompt
    prompt = f"""
You are a medical AI assistant specialized in hematology. Based on the following patient data, provide a concise but informative report.

Patient Information:
- Age: {patient_data['age']} years
- Gender: {patient_data['gender']}
- Hemoglobin: {patient_data['hemoglobin']} g/dL
- RBC count: {patient_data['rbc']} million/µL
- Hematocrit: {patient_data['hematocrit']}%
- MCV: {patient_data['mcv']} fL
- MCH: {patient_data['mch']} pg
- MCHC: {patient_data['mchc']} g/dL

Analysis:
- Anemia detected: {diagnosis['anemic']}
- Severity: {diagnosis['severity']}
- Morphology: {morphology}

Please include:
1. A brief explanation of the findings.
2. Potential risk factors or common causes associated with this pattern.
3. General lifestyle and dietary recommendations (if appropriate).
4. A disclaimer that this is not a substitute for professional medical advice.

Keep the tone empathetic and easy to understand.
"""
    try:
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",  # or "llama3-70b-8192"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return None

# ------------------------------------------------------------------------------
# UI Layout
# ------------------------------------------------------------------------------
st.title("🩸 HemoScan AI – Anemia Detection & Risk Analysis")
st.markdown("""
This application helps detect anemia based on standard blood parameters and provides 
an AI‑powered analysis using Groq's large language models.  
**Note:** This tool is for informational purposes only and does not replace professional medical advice.
""")

# Input section
with st.sidebar:
    st.header("Patient Information")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])

    st.header("Blood Parameters")
    hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=12.5, step=0.1)
    rbc = st.number_input("RBC count (million/µL)", min_value=0.0, max_value=10.0, value=4.5, step=0.1)
    hematocrit = st.number_input("Hematocrit (%)", min_value=0.0, max_value=60.0, value=38.0, step=0.1)
    mcv = st.number_input("MCV (fL)", min_value=50.0, max_value=150.0, value=90.0, step=0.1)
    mch = st.number_input("MCH (pg)", min_value=15.0, max_value=40.0, value=29.0, step=0.1)
    mchc = st.number_input("MCHC (g/dL)", min_value=25.0, max_value=40.0, value=33.0, step=0.1)

    analyze_button = st.button("Analyze", type="primary")

# Main panel
if analyze_button:
    # 1. Basic detection
    anemic, severity = detect_anemia(hemoglobin, gender, age)
    diagnosis = {"anemic": anemic, "severity": severity}

    # 2. Morphology classification (only meaningful if anemic, but we compute anyway)
    morphology = classify_morphology(mcv, mch, mchc)

    # 3. Display results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detection Result")
        if anemic:
            st.error(f"**Anemia Detected** – Severity: {severity}")
        else:
            st.success("**No Anemia Detected**")

        st.write(f"**Morphology (based on indices):** {morphology}")

    with col2:
        st.subheader("Input Summary")
        data = {
            "Parameter": ["Hemoglobin", "RBC", "Hematocrit", "MCV", "MCH", "MCHC"],
            "Value": [f"{hemoglobin} g/dL", f"{rbc} M/µL", f"{hematocrit}%", f"{mcv} fL", f"{mch} pg", f"{mchc} g/dL"]
        }
        st.table(pd.DataFrame(data))

    # 4. Generate Groq report
    st.subheader("AI‑Generated Risk Analysis")
    with st.spinner("Contacting Groq AI for insights..."):
        patient_data = {
            "age": age,
            "gender": gender,
            "hemoglobin": hemoglobin,
            "rbc": rbc,
            "hematocrit": hematocrit,
            "mcv": mcv,
            "mch": mch,
            "mchc": mchc
        }
        report = generate_groq_report(patient_data, diagnosis, morphology)
        if report:
            st.markdown(report)
        else:
            st.warning("Report could not be generated. Please check your API key and try again.")
else:
    st.info("👈 Enter patient data in the sidebar and click **Analyze** to start.")

# Footer
st.markdown("---")
st.caption("HemoScan AI – Powered by Streamlit and Groq. Always consult a healthcare professional for medical decisions.")
