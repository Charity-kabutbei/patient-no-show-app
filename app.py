import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")

st.set_page_config(page_title="Home", layout="wide")

# STYLE
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #f1f5f9;
}

/* HEADER BAR */
.header {
    background: linear-gradient(90deg, #667eea, #764ba2);
    padding: 15px 30px;
    border-radius: 10px;
    margin-bottom: 30px;
    display: flex;
    align-items: center;
}

/* HEADER TEXT */
.header-title {
    color: black;
    font-size: 30px;
    font-weight: bold;
    margin-left: 15px;
}

/* MAIN CARD */
.card {
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.1);
    max-width: 900px;
    margin: auto;
}

/* TITLE */
.title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 25px;
}

/* BUTTON */
.stButton>button {
    background: #667eea;
    color: white;
    border-radius: 8px;
    width: 100%;
    padding: 10px;
    font-size: 16px;
    border: none;
}

.stButton>button:hover {
    background: #5a67d8;
}

</style>
""", unsafe_allow_html=True)

#  HEADER
col1, col2 = st.columns([1,10])

with col1:
    st.image("logo.png", width=50)

with col2:
    st.markdown('<div class="header-title">Patient No-Show Predictor</div>', unsafe_allow_html=True)



#  CARD START
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 0, 100, 25)

with col2:
    sms = st.selectbox("SMS Received", ["No", "Yes"])
    waiting = st.number_input("Waiting Days", 0, 100, 3)

gender = 1 if gender == "Male" else 0
sms = 1 if sms == "Yes" else 0

st.markdown("<br>", unsafe_allow_html=True)

#  PREDICT
if st.button("Predict Appointment Outcome"):

    data = np.array([[gender, age, sms, waiting]])
    pred = model.predict(data)
    prob = model.predict_proba(data)

    st.markdown("---")

    if pred[0] == 1:
        st.error(f"⚠️ No-Show Risk: {prob[0][1]:.2%}")
    else:
        st.success(f"✅ Likely to Attend: {prob[0][0]:.2%}")

    st.subheader("Prediction Confidence")

    labels = ["Show", "No-Show"]
    values = [prob[0][0], prob[0][1]]

    plt.figure()
    plt.bar(labels, values)
    st.pyplot(plt)

# 🔵 CARD END
st.markdown('</div>', unsafe_allow_html=True)