# Campus Placement Predictor - Technical Documentation

## 1. Project Overview
The **Campus Placement Predictor** is an AI-driven decision support system designed to estimate the probability of a student getting placed based on their academic and professional profile. It also provides salary estimations and career roadmap recommendations. The project caters to two distinct domains: **MBA** and **Engineering**, using tailored data and logic for each.

## 2. Machine Learning Models (Core Focus)
The project utilizes a **Multi-Model Intelligence System** where users can switch between 4 different "brains" (classification algorithms) to compare performance and logic.

### Classification Models (Placement Prediction)
1.  **Gradient Boosting (GBM)**:
    - **Why?**: It builds trees sequentially, correcting errors of previous trees. It's highly accurate for complex tabular data where features have non-linear relationships.
2.  **Random Forest (RF)**:
    - **Why?**: An ensemble method that reduces variance by averaging multiple decision trees. It is robust to noise and less prone to overfitting than a single decision tree.
3.  **Logistic Regression (LR)**:
    - **Why?**: Provides a statistical baseline. It is highly interpretable and works exceptionally well for binary classification when data is scaled (as implemented in our pipeline).
4.  **Extra Trees (ET)**:
    - **Why?**: Similar to Random Forest but chooses split points randomly, making it faster and sometimes better at generalizing on synthetic or noisy data.

### Regression Models (Salary Prediction)
- **Random Forest Regressor**: Used for predicting continuous salary values for placed students.
- **Gradient Boosting Regressor**: Optimized for handling variance in salary packages.
- **Linear & Ridge Regression**: Used for stable, linear salary estimations.

## 3. Accuracy & Comparison
The models are evaluated using a **20% hold-out test set** and a **Global Confusion Matrix** representing all 800 samples.

| Model Type | MBA Accuracy | Engineering Accuracy | Characteristics |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | ~87-90% | ~76-80% | Best at capturing complex patterns. |
| **Random Forest** | ~88-92% | ~75-78% | Most stable across different datasets. |
| **Logistic Regression**| ~88-95% | ~74-76% | Highest interpretability. |
| **Extra Trees** | ~82-85% | ~77-79% | High variance reduction. |

### Why these accuracies?
- **MBA Data**: Shows higher accuracy because features like `workex` and `degree_p` are very strong predictors of placement.
- **Engineering Data**: Shows slightly lower accuracy (75-80%) reflecting the high volatility and diverse skill requirements in tech industries.

## 4. Dataset Details
The project uses two primary datasets, each expanded to **800 rows** using synthetic data generation to ensure statistical significance and model stability.

### MBA Dataset (`Placement_Data_Full_Class.csv`)
- **Features**: Gender, SSC Percentage, HSC Percentage, Degree Percentage, Work Experience, E-Test Score, Specialization, MBA Percentage.
- **Target**: Placement Status & Salary.

### Engineering Dataset (`Engineering.csv`)
- **Features**: Gender, 10th Marks, 12th Marks, Stream, CGPA, Technical Score, Internships, Projects, Backlogs.
- **Target**: Placement Status.

## 5. Technology Stack
- **Core Logic**: Python 3.x
- **Machine Learning**: Scikit-Learn (Ensemble methods, Linear models, Preprocessing)
- **Data Handling**: Pandas & NumPy
- **Web Interface**: Streamlit (Dashboard, Real-time Simulation, Forensics)
- **Visualization**: Matplotlib & Seaborn
- **Serialization**: Joblib (Model persistence)

## 6. Combatting Overfitting/Underfitting
To ensure a "Goldilocks" fit (neither too simple nor too complex), we implemented:
- **Regularization**: L2 penalty for Logistic Regression and `min_samples_leaf` for tree-based models.
- **Max Depth Limits**: Prevented trees from growing infinitely deep and memorizing noise.
- **Standard Scaling**: All numeric data is normalized before training to ensure features like `salary` don't overshadow percentages.
