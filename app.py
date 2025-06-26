import streamlit as st
import joblib
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from sklearn.preprocessing import MinMaxScaler

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")  # Isolation Forest or similar

# Load LLM
llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    api_key="1882eed17b84bbee420d26f95a1342d5453e16fb1adfbe9caf161e1136143d7f",
    max_tokens=600
)

# Streamlit UI
st.set_page_config(page_title="HVAC Fault Detector", layout="centered")
st.title("HVAC Anomaly Detection & Fault Diagnosis")
st.markdown("Enter real-time sensor data to detect faults.")

# Input fields
co2 = st.number_input("CO2 (ppm)", min_value=0.0, value=800.0)
humidity = st.number_input("Humidity(%)", min_value=0.0, max_value=100.0, value=50.0)
occupancy = st.number_input("Occupancy", min_value=0, value=5)
temp = st.number_input("Temperature (°C)", min_value=0.0, value=24.0)

if st.button("Analyze"):
    # Predict anomaly
    input_data = np.array([[temp,humidity,co2,occupancy]])
    score = model.decision_function(input_data)[0]
    score = scaler.transform([[score]])[0][0]

    st.subheader(f"Anomaly Score: {score:.2f}")
    if score < 0.5:
        st.error("⚠️ Critical Situation Detected! High likelihood of severe fault.")
    elif score < 0.7:
        st.warning("⚠️ Anomaly Detected! Please investigate the system.")
    else:
        st.success("✅ No anomaly detected. System operating normally.")

    # If anomaly, run LLM
    if score < 0.7:
        prompt = PromptTemplate(
            input_variables=["co2", "humidity", "occupancy", "temp"],
            template="""
            You are an HVAC System Fault Diagnosis Expert.

            The system has reported an anomaly.

            Given the following sensor values:
            - CO2: {co2} ppm
            - Humidity: {Humidity}
            - Occupancy: {occupancy}
            - Temperature: {temp} °C

            Please analyze and explain the most likely cause of this anomaly, 
            and suggest any necessary actions to resolve or further investigate the issue.
            """
        )

        chain = prompt | llm
        with st.spinner("Consulting HVAC knowledge base..."):
            response = chain.invoke({
                "co2": co2,
                "Humidity": humidity,
                "occupancy": occupancy,
                "temp": temp
            })

        st.subheader("📘 Suggested Fault Diagnosis:")
        st.markdown(response.content)
