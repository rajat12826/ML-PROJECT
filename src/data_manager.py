import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

class DataManager:
    def __init__(self, mode='MBA'):
        self.mode = mode
        # Reverting to the Logical Rewritten datasets
        self.file_path = 'Placement_Data_Full_Class.csv' if mode == 'MBA' else 'Engineering.csv'
        self.df = None
        self.encoders = {}

    def load_data(self):
        """Loads the logical rewritten datasets."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
            
        self.df = pd.read_csv(self.file_path)
        
        if self.mode == 'MBA':
            self.df['salary'] = self.df['salary'].fillna(0)
        else:
            # Engineering status is already 'status' in the rewritten file
            pass
            
        return self.df

    def preprocess_data(self, target_col='status'):
        """Encoded categorical variables and splits features/target."""
        if self.df is None:
            self.load_data()
            
        temp_df = self.df.copy()
        
        categorical_cols = temp_df.select_dtypes(include=['object']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
            
        # Standardize status
        temp_df['status'] = temp_df['status'].map({'Placed': 1, 'Not Placed': 0, 'Yes': 1, 'No': 0}).fillna(0)
        
        for col in categorical_cols:
            le = LabelEncoder()
            temp_df[col] = temp_df[col].fillna('Unknown').astype(str).str.strip().str.title()
            temp_df[col] = le.fit_transform(temp_df[col])
            self.encoders[col] = le
            
        temp_df.dropna(subset=[target_col], inplace=True)
            
        cols_to_drop = ['status', 'salary']
        X = temp_df.drop(cols_to_drop, axis=1, errors='ignore')
        y = temp_df['status']
        
        return X, y, self.encoders
