# 🎓 Campus Placement Predictor (Logic-First Edition)

A premium, state-of-the-art placement prediction system that uses **Gradient Boosting (GBM)** and **Real-World Logical Constraints** to provide accurate career guidance.

## 🚀 Key Features
*   **Intelligence Modes**: Dual support for Engineering and MBA career paths.
*   **Real-World Logic Engine**: Moves beyond simple academic history. Factors in **Backlogs**, **Technical Projects**, **Technical Scores**, and **Employability Tests**.
*   **Predictive Sensitivity**: Captures hard-floor failures (e.g., failing a core test results in 0.0% probability) using advanced GBM models.
*   **Model Forensics**: Deep-dive into AI accuracy, F1-Scores, and Confusion Matrices.
*   **Career Roadmaps**: Stream-specific skill recommendations and company tiering.

## 🛠️ Technology Stack
*   **Core**: Python 3.x
*   **Frontend**: Streamlit (Premium Glassmorphism UI)
*   **Machine Learning**: Scikit-Learn (GradientBoostingClassifier, RandomForestRegressor)
*   **Data Handling**: Pandas, NumPy
*   **Visualization**: Matplotlib, Seaborn

## 📂 Project Structure
*   `app.py`: The main premium UI and dashboard.
*   `src/data_manager.py`: Robust data cleaning and preprocessing.
*   `src/model_trainer.py`: Gradient Boosting training pipeline.
*   `src/analyzer.py`: Peer benchmarking and salary tiering logic.
*   `src/recommender.py`: Skills and career roadmap generator.
*   `models/`: Serialized models and performance metrics.

## 📈 Logical Constraints Verified
This model is designed to be **Logical**.
*   **Engineering**: High CGPA is ignored if you have >1 Backlog or Failing Technical Scores.
*   **MBA**: High 10th/12th marks are ignored if your **E-test** or **Degree %** is critically low.
*   **Result**: No more "biases" where bad students get predicted as placed.

---
**Developed by Antigravity AI**
