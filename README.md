# 🎓 Campus Placement Predictor

A premium Machine Learning application built with **Streamlit** to predict student placement probability and estimate salary packages for **MBA** and **Engineering** domains.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)

---

## 🚀 Features
- **Dual Domain Intelligence:** Specialized models for MBA and Engineering datasets.
- **NLP Resume Parser:** Upload a PDF resume to autofill profile details and extract technical skills.
- **Explainable AI:**
  - **Probability Score:** Real-time placement chance in %.
  - **Success Path Optimizer:** Interactive "What-If" analysis to see how score improvements affect results.
  - **Peer Benchmarking:** Compare performance against the average of placed students.
- **Career Roadmap:** Personalized career path recommendations based on domain and skills.
- **Premium UI:** Dark-mode "Linear" style aesthetics with glassmorphic cards and dynamic visualizations.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ml-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Models (Optional - Models are pre-trained):**
   ```bash
   python train_all.py
   ```

4. **Launch Application:**
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure
- `app.py`: Main Streamlit application.
- `src/`: Core logic modules.
  - `data_manager.py`: Data loading and preprocessing.
  - `model_trainer.py`: Model architecture and training pipeline.
  - `resume_parser.py`: NLP logic for PDF text extraction.
  - `recommender.py`: Rule-based career guidance.
  - `analyzer.py`: Benchmarking and tier classification.
  - `eda_utils.py`: Visualization styling and plotting.
- `models/`: Serialized `.pkl` files (Logistic Regression & Random Forest).
- `PROJECT_REPORT.md`: Detailed documentation for documentation/submission.
- `VIVA_CHEATSHEET.md`: Line-by-line explanation for viva preparation.

---

## 🧠 Machine Learning Overview
- **Classification:** Logistic Regression (for binary placement prediction).
- **Regression:** Random Forest Regressor (for salary estimation).
- **Split:** 80% Training | 20% Testing.
- **Preprocessing:** Label Encoding for categorical data and heuristic normalization.

---
*Developed for ML Lab Project* 🚀
