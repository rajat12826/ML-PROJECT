# 🗺️ System Architecture & Data Flow

This document provides a visual representation of how the Campus Placement Predictor works, from data loading to real-time prediction.

---

## 🔁 Overall System Pipeline

```mermaid
graph TD
    A[Raw Data: CSV Files] --> B[DataManager]
    B --> C{Preprocessing}
    C -->|Handle Missing Values| D[Cleaning]
    C -->|Label Encoding| E[Numeric Conversion]
    D & E --> F[Processed Features X / Target y]
    F --> G[ModelTrainer]
    G -->|80/20 Split| H[Training & Validation]
    H -->|Logistic Regression| I[Saving Placement Model]
    H -->|Random Forest| J[Saving Salary Model]
    I & J --> K[Streamlit App]
    L[User Input / Resume PDF] --> K
    K --> M[Final Prediction & Recommendations]
```

---

## 🛠️ Data Preprocessing Flow

```mermaid
flowchart LR
    Start([Load CSV]) --> FillNaN[Fill Salary NaN with 0]
    FillNaN --> DropCols[Drop sl_no, Name, Email]
    DropCols --> Rename[Standardize Column Names]
    Rename --> Encode[Label Encoding for text data]
    Encode --> Binary[Map Status: Placed=1, Not Placed=0]
    Binary --> Split[Return X and y]
```

---

## 🎯 Prediction & Recommendation Logic

```mermaid
sequenceDiagram
    participant User
    participant App
    participant Model
    participant Recommender

    User->>App: Input Details or Upload Resume
    App->>App: Parse Resume (PDF Text)
    App->>App: Encode Features
    App->>Model: predict_proba(X)
    Model-->>App: Return Probability (e.g. 0.85)
    App->>Model: predict_salary(X)
    Model-->>App: Return Estimated Package
    App->>Recommender: recommend_path(Data)
    Recommender-->>App: Return Career Roadmaps
    App-->>User: Display Final Insight Portal
```

---

## 📊 Dashboard & Analytics Flow

1.  **Exploratory Data Analysis (EDA):** Uses Seaborn and Matplotlib to generate real-time charts.
2.  **Benchmarking:** Calculates the mean of `ssc_p`, `hsc_p`, etc., from the **Placed** subset and calculates the student's percentile.
3.  **What-If Analysis:** Adjusts a single feature value, re-runs the model prediction, and calculates the `gain` in probability.

---
*Visual Guide for ML Project Architecture* 🚀
