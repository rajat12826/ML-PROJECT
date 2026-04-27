import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, 
    ExtraTreesClassifier, AdaBoostClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
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
        
        self.classification_models = {
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=4, 
                min_samples_leaf=5, random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=4, 
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, C=0.5, penalty='l2', random_state=42
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=4, 
                random_state=42
            )
        }
        
        self.regression_models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=50, max_depth=5, min_samples_leaf=3, 
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=3, 
                random_state=42
            ),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0)
        }
        self.scaler = StandardScaler()

    def train_classification(self, X, y, model_type='Gradient Boosting'):
        """Trains placement prediction model rapidly."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling for consistency
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model = self.classification_models.get(model_type, self.classification_models['Gradient Boosting'])
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Metrics (Test Set)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm_test = confusion_matrix(y_test, y_pred).tolist()
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        # Metrics (Full Set for Dashboard/Forensics)
        X_full_scaled = self.scaler.transform(X)
        y_pred_full = model.predict(X_full_scaled)
        cm_full = confusion_matrix(y, y_pred_full).tolist()
        
        importance = {}
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            importance = dict(zip(X.columns, np.abs(model.coef_[0])))

        return {
            "accuracy": acc,
            "f1_score": f1,
            "confusion_matrix": cm_test,
            "confusion_matrix_full": cm_full,
            "report": report,
            "feature_importances": importance,
            "model_obj": model,
            "sample_count": len(X)
        }

    def train_regression(self, X, y, model_type='Random Forest'):
        """Trains salary prediction model."""
        if self.mode != 'MBA' or len(X) == 0:
            return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use existing scaler fit or fit new one
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model = self.regression_models.get(model_type, self.regression_models['Random Forest'])
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {"rmse": rmse, "r2": r2, "model_obj": model}

    def save_models(self, features, encoders, classification_results, regression_results=None, model_name='default'):
        """Serializes models, features, encoders and metrics."""
        # Save classification model
        model_type_slug = model_name.lower().replace(" ", "_")
        joblib.dump(classification_results['model_obj'], f'{self.model_dir}/placement_model_{model_type_slug}.pkl')
        
        # Save regression model if available
        if regression_results:
            joblib.dump(regression_results['model_obj'], f'{self.model_dir}/salary_model_{model_type_slug}.pkl')
        
        # Save Scaler
        joblib.dump(self.scaler, f'{self.model_dir}/scaler.pkl')
        
        # Standard resources
        joblib.dump(features, f'{self.model_dir}/features.pkl')
        joblib.dump(encoders, f'{self.model_dir}/label_encoders.pkl')
        
        # Save specific metrics for this model
        metrics = {
            "classification": {k: v for k, v in classification_results.items() if k != 'model_obj'},
            "regression": {k: v for k, v in regression_results.items() if k != 'model_obj'} if regression_results else None
        }
        joblib.dump(metrics, f'{self.model_dir}/metrics_{model_type_slug}.pkl')
