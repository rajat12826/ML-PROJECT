import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import joblib
import os

class ModelTrainer:
    def __init__(self, mode='MBA'):
        self.mode = mode
        self.model_dir = f'models/{mode.lower()}'
        os.makedirs(self.model_dir, exist_ok=True)
        self.classification_model = LogisticRegression(max_iter=1000)
        self.regression_model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train_classification(self, X, y):
        """Trains placement prediction model."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classification_model.fit(X_train, y_train)
        y_pred = self.classification_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        return report

    def train_regression(self, X, y):
        """Trains salary prediction model (MBA only)."""
        if self.mode != 'MBA':
            return None
        # For regression, we only train on placed students
        # Note: X and y passed here should already be filtered for PLACED by the caller if it's regression
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.regression_model.fit(X_train, y_train)
        y_pred = self.regression_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return {"mse": mse, "r2": self.regression_model.score(X_test, y_test)}

    def save_models(self, features, encoders):
        """Serializes models, features, and encoders."""
        joblib.dump(self.classification_model, f'{self.model_dir}/placement_model.pkl')
        if self.mode == 'MBA':
            joblib.dump(self.regression_model, f'{self.model_dir}/salary_model.pkl')
        joblib.dump(features, f'{self.model_dir}/features.pkl')
        joblib.dump(encoders, f'{self.model_dir}/label_encoders.pkl')
