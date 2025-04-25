# 🔐 AI-Powered Ransomware Detection using Memory Forensics

This project leverages machine learning and memory forensic analysis to detect ransomware from volatile memory dumps. We use advanced classification models including **Logistic Regression**, **Random Forest**, and **LightGBM**, trained on features extracted using the **Volatility** framework. Our goal is to create a proactive detection system that identifies ransomware behavior at the memory level.

---

## 📌 Objective

To build an AI-powered solution that accurately detects ransomware based on forensic artifacts found in memory, enabling better response and mitigation strategies before ransomware fully executes.

---

## 📁 Dataset

**Dataset Name:** [Windows Malware Detection using Volatility Memory Forensics Features](https://www.kaggle.com/datasets/azizmohamed/windows-malware-detection-using-volatility-features)

- **Total Records:** 531
- **Features:** 529 memory forensics-based features
- **Target Columns:**
  - `Class`: Indicates Malware (1) or Benign (0)
  - `Category`: Malware category (e.g., Trojan, Worm, Ransomware)
  - `Family`: Specific malware family (e.g., Locky, Zeus)

The dataset is created using forensic plugins in Volatility such as:
- `pslist`, `psscan`, `dlllist`, `handles`, `malfind`, `svcscan`, etc.

These help in identifying abnormal processes, injected code, and suspicious DLLs – all indicators of ransomware activity.

---

## 🧠 Models Used

We implemented and evaluated three machine learning models:

1. **Logistic Regression** – Simple linear classifier for benchmarking
2. **Random Forest Classifier** – Ensemble model offering better performance and interpretability
3. **LightGBM** – Gradient boosting model which provided the **best results**

---

## ⚙️ Tech Stack

- **Python**
- **Jupyter Notebook**
- **Scikit-learn**
- **LightGBM**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **Volatility (for dataset creation)**

---

## 🧪 Evaluation Metrics

We used the following metrics to evaluate model performance:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 📊 Results Summary:

| Model              | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | ~92%     | Moderate  | Moderate | Moderate |
| Random Forest       | ~98%     | High      | High     | High     |
| LightGBM            | **~99%** | **Very High** | **Very High** | **Very High** |

---

## 📂 Code Structure

```plaintext
ai-powered-ransomware-detection/
│
├── dataset/
│   └── memory_forensics.csv
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── lightgbm_model.txt
│
├── visualizations/
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── ransomware_detection.ipynb
├── requirements.txt
└── README.md
