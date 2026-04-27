# 🚶 Code Walkthrough: Logical Placement Predictor

This guide explains the step-by-step logic behind the new **Logic-First** prediction engine.

### 1. Data Ingestion (`data_manager.py`)
Incoming data is sanitized using a **Unified Cleaning Overhaul**. 
*   **Typos Fixed**: Automatic mapping of `Yess`, `Noo`, `Y` -> `Yes`, `No`.
*   **Target Standardization**: Status strings are mapped to binary integers (1 for Placed, 0 for Not Placed).
*   **Missing Values**: Handled using 'Unknown' labels or median imputation for scores.

### 2. The Logic-First Dataset
Unlike standard datasets that are biased towards early school marks, our datasets (`Engineering.csv` and `Placement_Data_Full_Class.csv`) include **Hard-Floor Rules**:
*   **Technical Floor**: If Tech Score < 45, Placement = 0.
*   **Stability Floor**: If Backlogs > 1, Placement = 0.
*   **Aptitude Floor**: If E-test < 40, Placement = 0.

### 3. Model Training (`model_trainer.py`)
We use **Gradient Boosting (GBM)**.
*   **Why GBM?**: GBM builds trees sequentially, focusing on the hardest-to-predict cases (like students with good grades but bad skills).
*   **Sensitivity**: GBM is much more sensitive to "Negative Indicators" compared to Random Forest or Logistic Regression.

### 4. Dynamic UI Logic (`app.py`)
The application is **Dataset-Agnostic**.
*   **Form Generation**: The input form is generated dynamically from the trained `features.pkl`. If the dataset changes, the UI updates automatically.
*   **Sensitivity Patch**: Custom code scans for "Critical Skills Gaps" and alerts the user even if the model's historical confidence is high.

### 5. Recommendation Engine (`recommender.py`)
Based on the `Stream` and `CGPA`, it suggests specific industrial roles (e.g., Investment Banking for Finance MBA, SDE for CSE Engineering) and lists the required modern tech stack.

---
**Summary**: This isn't just a basic ML script; it's a diagnostic tool that enforces industrial hiring standards.
