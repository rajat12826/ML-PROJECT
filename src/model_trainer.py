import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    mean_squared_error, r2_score
)
import joblib
import os

class ModelTrainer:
    def __init__(self, mode='MBA'):
        self.mode = mode
        self.model_dir = f'models/{mode.lower()}'
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Fast defaults instead of long GridSearchCV
        self.classification_model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
        self.regression_model = RandomForestRegressor(n_estimators=50, random_state=42)

    def train_classification(self, X, y):
        """Trains placement prediction model rapidly."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.classification_model.fit(X_train, y_train)
        y_pred = self.classification_model.predict(X_test)
        
        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        return {
            "accuracy": acc,
            "f1_score": f1,
            "confusion_matrix": cm,
            "report": report,
            "feature_importances": dict(zip(X.columns, self.classification_model.feature_importances_))
        }

    def train_regression(self, X, y):
        """Trains salary prediction model."""
        if self.mode != 'MBA' or len(X) == 0:
            return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.regression_model.fit(X_train, y_train)
        
        y_pred = self.regression_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {"rmse": rmse, "r2": r2}

    def save_models(self, features, encoders, metrics=None):
        """Serializes models, features, encoders and metrics."""
        joblib.dump(self.classification_model, f'{self.model_dir}/placement_model.pkl')
        if self.mode == 'MBA':
            joblib.dump(self.regression_model, f'{self.model_dir}/salary_model.pkl')
        joblib.dump(features, f'{self.model_dir}/features.pkl')
        joblib.dump(encoders, f'{self.model_dir}/label_encoders.pkl')
        if metrics:
            joblib.dump(metrics, f'{self.model_dir}/metrics.pkl')
