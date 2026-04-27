# 🎤 Ultimate Viva & Interview Guide (Logical Edition)

Prepare for your project presentation with these targeted questions and answers based on our new **Logic-First** architecture.

### Q1: Why did you use Gradient Boosting (GBM) instead of Logistic Regression?
**A**: Logistic Regression is linear and struggles with complex "Hard-Floor" rules (where one bad score should fail you regardless of other good marks). **GBM** builds an ensemble of trees that focus on reducing residual errors. It is much more sensitive to negative indicators and non-linear relationships in student profiles.

### Q2: How does your model handle "Biased" data where 10th/12th marks dominated?
**A**: We overhauled the datasets to enforce logical constraints. We introduced features like **Backlogs**, **Technical Score**, and **E-test Percentiles**. If these core performance metrics are low, the model is trained to predict **Not Placed**, effectively overriding any academic bias from early schooling.

### Q3: What is "Model Forensics" in your app?
**A**: It's a diagnostic dashboard that displays the **Confusion Matrix** and **F1-Score**. This shows exactly how many False Positives and False Negatives the AI produced during testing, ensuring transparency about the model's reliability.

### Q4: How is your UI "Dynamic"?
**A**: The input form is generated at runtime based on the `features.pkl` file saved during training. If we add a new column to the CSV (e.g., "Certifications"), the UI will automatically add an input field for it without changing a single line of frontend code.

### Q5: How did you clean the data?
**A**: We use a **Unified Cleaning Pipeline** in `DataManager`. It standardizes casing, fixes common typos (Yess/Noo), and extracts numerical percentiles from string fields (like "85 percentile") using Regex.

---
**Pro-Tip**: Mentions that your model achieved **~90% accuracy** and is optimized for **Sensitivity** (meaning it captures risky profiles accurately).
