# 🛡️ Project Viva Cheat Sheet: Line-by-Line Logic

This guide provides a technical breakdown of the core project files. Use this to explain "what each block of code does" to an examiner.

---

## 1. `app.py` (The Brain of the UI)
*   **Imports (Lines 1-11):** We import `streamlit` for the UI, `joblib` to load our pre-trained models, and our custom modules from the `src/` folder.
*   **Page Config (Line 14):** Sets the browser tab title and favicon.
*   **Custom CSS (Lines 17-139):** This is where the magic happens. We use HTML/CSS to inject a dark theme, glowing "blobs" for background aesthetics, and custom fonts (Inter).
*   **`load_resources` (Lines 150-168):** Uses `@st.cache_resource` so models are only loaded into RAM **once**. If the user switches from "MBA" to "Engineering", it loads the respective `.pkl` files.
*   **`init_session_state` (Lines 172-195):** Ensures that when the app starts, the input fields have default values and don't crash when switching modes.
*   **`show_results` (Lines 343-411):** 
    - First, it encodes the user's input using the saved `LabelEncoder`.
    - Then, it calls `predict_proba()` to get the placement percentage.
    - If placed, it calls the `salary_model` to predict the package.
*   **Success Path Optimizer (Lines 414-439):** A "What-If" tool. It takes the current input, adds a "boost" (e.g., +5% CGPA), and runs the prediction again to show the gain.

---

## 2. `src/data_manager.py` (The Data Scientist)
*   **`load_data` (Lines 13-32):**
    - Loads CSV files using Pandas.
    - **MBA Logic:** Fills missing `salary` with 0 (cleanliness).
    - **Engineering Logic:** Renames columns to match our internal naming convention.
*   **`preprocess_data` (Lines 34-59):**
    - **Line 41:** Identifies categorical columns (text-based).
    - **Line 46:** Maps status "Placed" to 1 and "Not Placed" to 0.
    - **Line 49 (LabelEncoder):** This is the core. It loops through each text column and converts categories to numbers (e.g., Science=2, Arts=0, Commerce=1).

---

## 3. `src/model_trainer.py` (The Teacher)
*   **`__init__` (Lines 10-16):** Initializes two models:
    - **Classification:** `LogisticRegression` (Linear model for binary classification).
    - **Regression:** `RandomForestRegressor` (Ensemble model for predicting continuous salary values).
*   **`train_classification` (Lines 18-24):**
    - **Line 20 (The Split):** Splitting data into 80% Training and 20% Testing.
    - **Line 21 (`fit`):** The actual "learning" step where the model finds weights for features.
*   **`save_models` (Lines 38-44):** Uses `joblib.dump` to save the "learned" brain of our models so we don't have to retrain them every time.

---

## 4. `src/resume_parser.py` (The NLP Reader)
*   **`skill_map` (Lines 7-14):** A dictionary mapping job domains to relevant technical keywords.
*   **`extract_text_from_pdf` (Lines 16-25):** Uses `PdfReader` to loop through all pages of an uploaded PDF and convert them into one giant string.
*   **`identify_skills` (Lines 27-41):** Uses `re.search` (Regex) to find if any of our mapped keywords (like "Python") exist in the lowercased resume text.
*   **`extract_features` (Lines 43-84):** 
    - Uses regex `\d{1,2}\s*%` to find numeric patterns that look like percentages.
    - It intelligently guesses which percentage is SSC, HSC, or Degree based on their values (usually SSC/HSC are higher or first).

---

## 🎯 Quick Technical Keywords for Viva:
1.  **Overfitting:** We prevent this by dropping IDs (`sl_no`) and using a clean train-test split.
2.  **Sigmoid Function:** The math behind Logistic Regression that squashes output between 0 and 1.
3.  **Multicollinearity:** We check this using the Correlation Matrix (Heatmap) in the Dashboard.
4.  **Hyperparameters:** `n_estimators=100` in Random Forest controls how many trees are built.
5.  **Handling Imbalanced Data:** We use `classification_report` (Precision/Recall) instead of just Accuracy to ensure the model is fair to both Placed and Not Placed categories.

---
*Good luck with your Viva!* 🎓
