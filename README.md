# 🌱 AuraClima: AI-Powered Climate Forecasting with Agricultural Insights

**AuraClima** is a deep learning-driven climate prediction platform designed to forecast CO₂ emissions per country based on agricultural and economic indicators. Leveraging powerful LSTM and ANN models, AuraClima not only visualizes historic and future emission trends but also allows users to simulate changes using interactive sliders. Deployed on [Hugging Face Spaces](https://huggingface.co/spaces/AbdullahImran/AuraClima) via Streamlit, this tool bridges data science with climate activism.

---

## 🌍 Overview

AuraClima is built to assist:

* NGOs and policy makers
* Climate scientists and sustainability experts
* Data analysts
* General public passionate about environmental change

Our mission is to **inform decisions** through accurate CO₂ forecasts based on agricultural data from 1960 to 2018, projecting into 2028.

---

## 🧠 Core Features and Models

### 1️⃣ LSTM – Country-Wise CO₂ Forecast (1960–2028)

* **Model**: Deep LSTM architecture
* **Input**: Agricultural data from 1960 to 2018
* **Output**: Future CO₂ emissions for 2019–2028
* **Usage**: Time-series forecasting per country with dynamic line plots
* **Impact**: Reveals potential emission trajectories for proactive decision-making

---

### 2️⃣ LSTM – Agricultural CO₂ Trend Predictor

* **Model**: Country-wise LSTM
* **Input**: Historical agricultural indicators
* **Output**: Predicted CO₂ emissions specifically from agriculture
* **Usage**: Visualize agriculture’s impact on environment over decades

---

### 3️⃣ ANN – Interactive Multivariate Regressor

* **Model**: Feedforward neural network
* **Input**: User-adjustable sliders for variables like fertilizer use, livestock counts, land area, etc.
* **Output**: Instant prediction of adjusted CO₂ emissions
* **Usage**: Simulate “what-if” scenarios and policy decisions

---

## 🚀 Deployment

AuraClima is live and hosted on:

* **🟣 Hugging Face Spaces**: [AuraClima Space](https://huggingface.co/spaces/AbdullahImran/AuraClima)
* **🌐 Streamlit App**: Embedded inside Hugging Face
* **📦 GitHub Repo**: [View Source Code](https://github.com/poetabdullah/AuraClima)

---

## 📊 Charts & Marketing Materials (for academic evaluation)

| Chart | Description                                                               |
| ----- | ------------------------------------------------------------------------- |
| **1** | Product Analysis (core, augmented, actual), brand identity, target market |
| **2** | TV/media analysis for ad placement strategy                               |
| **3** | Ad script + pricing strategy (skimming model)                             |
| **4** | Storyboard for ad concept (shot using “Love for Mother Nature” theme)     |

📺 **Ad Video**: [Watch Ad on YouTube](https://www.youtube.com/watch?v=FJF5sx0S3R8)

---

## 💡 Impact

* ✅ Offers real-time simulation of agricultural policies and CO₂ effects
* ✅ Democratizes climate forecasting tools for public, NGOs, and analysts
* ✅ Empowers decision-makers with long-term projections
* ✅ Fully open-source and transparent

---

## 🛠 Tech Stack

* `Python`, `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
* `TensorFlow` & `Keras` for LSTM + ANN models
* `Streamlit` for frontend UI
* `Hugging Face Spaces` for deployment
* `scikit-learn` for preprocessing
* `Plotly` for interactive visualizations

---

## 📂 File Structure

```
AuraClima/
├── models/               # Trained ANN and LSTM models
├── notebooks/            # Development notebooks
├── streamlit_app.py      # Main app entry point
├── requirements.txt      # Python dependencies
├── README.md             # You are here!
```

---

## 📈 Sample Forecasts

* **India**: 18% projected increase in agri-CO₂ by 2028
* **Pakistan**: Plateau trend post-2022 with reduction possible under livestock control
* **USA**: Declining trend due to reduced agri intensity

---

## 🧑‍💻 Author

**Abdullah Imran**
[LinkedIn](https://www.linkedin.com/in/abdullah--imran/) | [GitHub](https://github.com/poetabdullah)

---

## 📜 License

MIT License. Free to use and build upon.
