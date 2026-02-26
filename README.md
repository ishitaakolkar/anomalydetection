# 🛒 Retail Insights AI: User Guide

Welcome to your **Retail Insights AI Explorer**! This tool uses cutting-edge Artificial Intelligence (Nixtla TimeGPT) to help you understand your sales patterns, catch unusual events, and predict the future—all without requiring you to be a data scientist.

---

## 🌟 What does this app do?

This dashboard transforms your raw sales spreadsheets into a live intelligence center. It focuses on three things:
1.  **Anomaly Detection**: Spotting "out-of-the-ordinary" sales days (Spikes or Dips).
2.  **Future Forecasting**: Predicting what your sales will look like for the next 7 days.
3.  **Automated Strategy**: Explaining *why* an event was unusual and suggesting business actions.

---

## 🧠 How the AI Works ("The Brain")

Traditional systems just look for "high numbers." This app is smarter. It uses **Zero-Shot AI**, which means it has been trained on billions of data points to understand:

*   **Seasonality**: It knows that a busy Saturday is normal, but a busy Tuesday is a "Spike."
*   **Trends**: It understands if your brand is growing and won't flag a healthy growth trend as an "error."
*   **Probability**: The AI draws a "Confidence Tube" around your data. If a sales point falls outside this tube, it is flagged as an anomaly.

---

## 📊 Understanding the Charts

When you select a category, you'll see a graph with three main components:

*   **🔵 Solid Blue Line**: Your actual historical sales.
*   **🚨 Red "X" Markers**: These are the **Anomalies**. The AI has flagged these points as mathematically impossible to explain through normal patterns.
*   **🟢 Dashed Green Line**: The AI's **7-Day Projection**. It shows where your sales are likely headed.
*   **🌫️ Light Green Shadow**: The "Confidence Interval." As long as your future sales stay inside this shadow, things are going according to plan.

---

## 💡 Automated Business Insights

At the bottom of the page, the AI generates **Insight Cards** for the most recent anomalies:

*   **🚀 Positive Spikes**: Sales were significantly *higher* than the AI expected. 
    *   *Example Strategy:* "Check for viral social media posts or successful promotions."
*   **📉 Negative Dips**: Sales were significantly *lower* than the AI's "pessimistic" floor. 
    *   *Example Strategy:* "Check for stock-outs, competitor price wars, or technical issues."

---

## 🛠️ How to Use the App

### 1. API Configuration
On the left sidebar, enter your **Nixtla API Key**. This is the "fuel" for the AI. If you don't have one, the app will show a warning.

### 2. Upload Your Data
You can use the default `retail_sales.csv` or upload your own CSV file. The app is **Universal**, meaning it can work with any time-series data.

### 3. Column Mapping
If you upload a new file, tell the AI which column is which:
*   **Date (ds)**: When the sale happened.
*   **Value (y)**: The dollar amount or quantity.
*   **Category/ID (unique_id)**: The product name, mall name, or department.

### 4. Adjust Sensitivity
*   **99.0% (Default)**: Only flags very rare, extreme events.
*   **90.0%**: More sensitive; will flag smaller deviations from the norm.

---

## 🛡️ Stability & Data Quality
*   **6-Month Window**: The app focuses on the most recent 180 days to ensure the AI's "memory" is fresh and relevant.
*   **Automatic Padding**: If a product has days with zero sales, the app automatically fills those gaps so the AI doesn't get confused.

---

**Ready to explore?** Upload your data and let the AI find the hidden stories in your sales! 🚀
