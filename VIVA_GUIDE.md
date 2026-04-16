# 🎓 Project Viva Guide: Campus Placement Predictor

This document provides a comprehensive explanation of the project structure, file-by-file logic, and potential questions you might face during your Viva/Oral examination.

---

## 📂 Project Structure & File Explanations

### 1. Root Directory
| File | Purpose | Key Responsibilities |
| :--- | :--- | :--- |
| **`app.py`** | **The Main Application** | The entry point. It manages the Streamlit UI, page routing, form handling, and coordinates all sub-modules (Model, Parser, Recommender). |
| **`train_all.py`** | **Automation Script** | A standalone script used to train both MBA and Engineering models from scratch and save them to the `models/` folder. |
| **`requirements.txt`** | **Dependencies** | Lists all Python libraries needed (`streamlit`, `scikit-learn`, `pypdf`, `pandas`, etc.). |
| **`PROJECT_REPORT.md`** | **Documentation** | A detailed report on the project's objectives, methodology, and results. |

### 2. Source Code (`src/`)
| File | Component | Logic Breakdown |
| :--- | :--- | :--- |
| **`data_manager.py`** | **Data Handler** | Loads CSVs, renames columns for consistency, and performs **Label Encoding** (converting text categories like "Science" to numbers like 2). |
| **`model_trainer.py`** | **ML Engine** | Defines models: `LogisticRegression` for Placement Prediction and `RandomForestRegressor` for Salary Prediction. Handles the 80-20 Train-Test split. |
| **`resume_parser.py`** | **NLP Parser** | Uses `PdfReader` to extract text from resumes and **Regex** (Regular Expressions) to find percentages and technical skills. |
| **`recommender.py`** | **Career Advisor** | A rule-based system that suggests job roles (e.g., "Data Scientist") based on your degree, skills detected in your resume, and academic interest. |
| **`analyzer.py`** | **Logic Layer** | Compares student scores against the average of placed students (**Benchmarking**) and categorizes salary into **Company Tiers** (Tier 1 vs Tier 3). |
| **`eda_utils.py`** | **Visualization** | Contains custom Matplotlib/Seaborn functions to plot Heatmaps and Dist charts with the "Linear Dark" aesthetic. |

### 3. Data & Models
| Folder/File | Purpose |
| :--- | :--- |
| **`*.csv`** | Raw datasets for Engineering & MBA placements. |
| **`models/`** | Stores `.pkl` files. These are "frozen brains" of our models so the app doesn't have to re-train every time it starts. |

---

## ❓ Potential Viva Questions & Answers

### 🔹 Category 1: General Logic & Programming
**Q1: Why did you choose Streamlit for the UI?**
*   **A:** Streamlit is a Python-based framework specifically designed for ML apps. It allows for rapid prototyping, handles session states (form inputs), and supports complex visualizations without needing HTML/CSS/JS expertise (though we used custom CSS for look & feel).

**Q2: What are `.pkl` files and why are they used?**
*   **A:** These are "Pickle" files. We use `joblib` to serialize (save) trained models and encoders. This is crucial because training takes time; once saved, we can load them instantly in the production app.

**Q3: How does your resume parser work?**
*   **A:** It uses the `pypdf` library to read PDF pages. Then, it uses **Regular Expressions (Regex)** to search for patterns. For example, `\d{1,2}\s*%` finds numbers followed by a percent sign. It also looks for keywords from a predefined skill dictionary.

---

### 🔸 Category 2: Machine Learning Concepts
**Q4: Which algorithm is used for placement prediction and why?**
*   **A:** We use **Logistic Regression**. It is a classification algorithm that outputs a probability between 0 and 1. It's efficient for binary outcomes (Placed vs. Not Placed) and highly interpretable.

**Q5: Why did you use Random Forest for Salary prediction?**
*   **A:** Salary is a continuous numeric value (Regression). Random Forest is an "Ensemble" method that combines multiple decision trees, making it robust against outliers and non-linear relationships in salary data.

**Q6: What is a Train-Test Split? What ratio did you use?**
*   **A:** We used an **80-20 split**. 80% of the data is used to "teach" the model patterns, and 20% is held back to "test" how well the model performs on data it has never seen before.

**Q7: How did you handle Categorical Data (Text)?**
*   **A:** Machine Learning models only understand numbers. We used **Label Encoding** from `scikit-learn` to convert text (e.g., "Commerce", "Science") into unique integers.

---

### 🚀 Category 3: Advanced Features & Edge Cases
**Q8: What is the "Success Path Optimizer"?**
*   **A:** It's a "What-If" analysis tool. It allows the student to virtually increase their scores (e.g., "What if I get 5% more CGPA?") and see the real-time boost in their placement probability.

**Q9: How do you handle "Out-of-Distribution" or unrealistic inputs (e.g., 110% marks)?**
*   **A:** We implement **Clamping**. We cap numeric inputs at realistic maximums (like 95% or 9.5 CGPA) before passing them to the model. This ensures the output remains stable and doesn't behave erratically for extreme values.

**Q10: Does the resume impact the ML prediction directly?**
*   **A:** Yes, we implemented **Skill Alignment Logic**. If a student selects "MBA Mode" but their resume only has "Java/Python" without any business skills, we apply a small penalty (e.g., 0.85x) to the model's output to make the result more realistic for that role.

---

## 💡 Pro Tips for your Viva:
1.  **Be Honest About Data:** If asked about accuracy, say "The model works well on the provided dataset, but in the real world, factors like interview performance (which we can't measure with CSVs) also matter."
2.  **Highlight Code Modularity:** Mention how you separated logic (`src/`) from UI (`app.py`). This shows you understand **good software engineering practices**.
3.  **Know your Accuracy:** If you ran `train_all.py` recently, check the logs for your Model's Accuracy/F1-Score and keep those numbers in mind.

---
*Created by Antigravity (Advanced Agentic Coding AI)*
