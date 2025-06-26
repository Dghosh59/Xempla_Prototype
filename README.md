# HVAC Anomaly Detection & Fault Diagnosis App

This Streamlit web app detects anomalies in HVAC (Heating, Ventilation, and Air Conditioning) systems using real-time sensor data and provides intelligent fault diagnosis powered by LLMs.

## 🔍 Features

- ✅ Real-time anomaly detection using a trained ML model (e.g., Isolation Forest).
- 🧠 Intelligent fault diagnosis using **LLaMA-3.3-70B-Instruct** via **Together.ai**.
- 📊 Inputs: CO₂ (ppm), Humidity (%), Occupancy, Temperature (°C).
- 📘 Contextual analysis and recommendations using LLMs when anomalies are found.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **ML Model**: Scikit-learn (Isolation Forest)
- **LLM**: [Together.ai](https://www.together.ai/) API (`meta-llama/Llama-3.3-70B-Instruct-Turbo-Free`)
- **Vector Store**: ChromaDB (placeholder for knowledge base expansion)
- **Embeddings**: HuggingFace Embeddings

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hvac-fault-detector.git
cd hvac-fault-detector
