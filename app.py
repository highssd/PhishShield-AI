
import streamlit as st
import pandas as pd
import joblib
from urllib.parse import urlparse
from datetime import datetime

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="PhishShield AI",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.title {
    font-size: 38px;
    font-weight: bold;
    color: #173b67;
}
.subtitle {
    color: #555;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD VERSION 2 MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("models/url_classifier_v2.pkl")
    vectorizer = joblib.load("models/url_vectorizer_v2.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.error(str(e))
    st.stop()

# -----------------------------------
# SESSION STATE
# -----------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------
# HEADER
# -----------------------------------

st.markdown(
    '<div class="title">🛡️ PhishShield AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Malicious URL Detection System</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Analyze a URL using machine learning and identify "
    "patterns associated with benign, phishing, malware "
    "and defacement URLs."
)

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("🛡️ PhishShield AI")

st.sidebar.write("Version 2")
st.sidebar.write("Algorithm: TF-IDF + Logistic Regression")

st.sidebar.markdown("---")

st.sidebar.subheader("Detection Categories")
st.sidebar.write("🟢 Benign")
st.sidebar.write("🔴 Phishing")
st.sidebar.write("🟠 Malware")
st.sidebar.write("🟡 Defacement")

st.sidebar.markdown("---")

st.sidebar.caption(
    "This application performs offline URL text analysis. "
    "It does not visit websites or verify their live safety."
)

# -----------------------------------
# DASHBOARD METRICS
# -----------------------------------

def calculate_metrics():
    total = len(st.session_state.history)

    threats = sum(
        1 for item in st.session_state.history
        if item["Prediction"] != "benign"
    )

    benign = total - threats

    return total, threats, benign


total_scans, threat_scans, benign_scans = calculate_metrics()

col1, col2, col3 = st.columns(3)

total_metric = col1.empty()
threat_metric = col2.empty()
benign_metric = col3.empty()

total_metric.metric("Total Scans", total_scans)
threat_metric.metric("Threat Predictions", threat_scans)
benign_metric.metric("Benign Predictions", benign_scans)

st.markdown("---")

# -----------------------------------
# URL SCANNER
# -----------------------------------

st.subheader("🔍 URL Security Scanner")

url = st.text_input(
    "Enter URL to scan",
    placeholder="https://example.com"
)

scan_button = st.button(
    "🔍 Scan URL",
    use_container_width=True
)

if scan_button:

    if not url.strip():
        st.warning("Please enter a URL.")

    else:
        entered_url = url.strip()

        if "://" not in entered_url:
            entered_url = "https://" + entered_url

        parsed = urlparse(entered_url)

        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or " " in entered_url
        ):
            st.error("Please enter a valid HTTP or HTTPS URL.")

        else:
            try:
                # Extract TF-IDF features
                transformed_url = vectorizer.transform(
                    [entered_url]
                )

                # Predict category
                prediction = model.predict(
                    transformed_url
                )[0]

                # Get model scores
                probabilities = model.predict_proba(
                    transformed_url
                )[0]

                score = max(probabilities) * 100

                # Save scan information
                scan_result = {
                    "Date and Time": datetime.now().strftime(
                        "%d-%m-%Y %H:%M:%S"
                    ),
                    "URL": entered_url,
                    "Prediction": prediction,
                    "Model Score (%)": round(score, 2)
                }

                st.session_state.history.append(scan_result)

                # UPDATE DASHBOARD METRICS IMMEDIATELY
                total_scans, threat_scans, benign_scans = calculate_metrics()

                total_metric.metric("Total Scans", total_scans)
                threat_metric.metric("Threat Predictions", threat_scans)
                benign_metric.metric("Benign Predictions", benign_scans)

                # -----------------------------------
                # SCAN RESULT
                # -----------------------------------

                st.markdown("---")
                st.subheader("📊 Scan Result")

                result_col1, result_col2 = st.columns(2)

                result_col1.write("**Scanned URL**")
                result_col1.write(entered_url)

                result_col2.write("**Predicted Category**")
                result_col2.write(prediction.upper())

                if prediction == "benign":
                    st.success("Model prediction: BENIGN")

                elif prediction == "phishing":
                    st.error(
                        "Warning: Model prediction indicates PHISHING."
                    )

                elif prediction == "malware":
                    st.error(
                        "Warning: Model prediction indicates MALWARE."
                    )

                elif prediction == "defacement":
                    st.warning(
                        "Warning: Model prediction indicates DEFACEMENT."
                    )

                st.metric(
                    "Model Score",
                    f"{score:.2f}%"
                )

                st.progress(
                    min(max(int(score), 0), 100)
                )

                st.caption(
                    "The model score is not a verified probability "
                    "that the URL is malicious or safe."
                )

                # -----------------------------------
                # CLASS SCORES
                # -----------------------------------

                st.subheader("Class Scores")

                for category, probability in zip(
                    model.classes_, probabilities
                ):
                    st.write(
                        f"**{category.capitalize()}:** "
                        f"{probability * 100:.2f}%"
                    )

                # -----------------------------------
                # URL INFORMATION
                # -----------------------------------

                st.subheader("🌐 URL Information")

                st.write("**Hostname:**", parsed.hostname)
                st.write("**Protocol:**", parsed.scheme)
                st.write("**Path:**", parsed.path or "/")

            except Exception as e:
                st.error("Error while scanning URL.")
                st.error(str(e))

# -----------------------------------
# SCAN HISTORY
# -----------------------------------

st.markdown("---")
st.subheader("📜 Scan History")

if st.session_state.history:

    for index, item in enumerate(
        reversed(st.session_state.history), start=1
    ):
        st.write(f"**Scan {index}**")
        st.write("URL:", item["URL"])
        st.write("Prediction:", item["Prediction"].upper())
        st.write(
            "Model Score:",
            str(item["Model Score (%)"]) + "%"
        )
        st.write("Date:", item["Date and Time"])
        st.markdown("---")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Scan History CSV",
        data=csv_data,
        file_name="phishshield_scan_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    if st.button("Clear Scan History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No URLs have been scanned yet.")

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.caption(
    "PhishShield AI | Cybersecurity Project | "
    "Machine Learning Based URL Detection"
)

st.caption(
    "Educational prototype only. A benign prediction does not "
    "guarantee that a website is safe. Always verify suspicious links."
)
