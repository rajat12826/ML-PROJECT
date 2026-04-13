import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

class DataManager:
    def __init__(self, mode='MBA'):
        self.mode = mode
        self.file_path = 'Placement_Data_Full_Class.csv' if mode == 'MBA' else 'Engineering.csv'
        self.df = None
        self.encoders = {}

    def load_data(self):
        """Loads and cleans the dataset based on current mode."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
            
        self.df = pd.read_csv(self.file_path)
        
        if self.mode == 'MBA':
            # Cleaning for MBA
            self.df['salary'] = self.df['salary'].fillna(0)
            self.df.drop(['sl_no'], axis=1, inplace=True, errors='ignore')
        else:
            # Cleaning for Engineering
            self.df.drop(['Email', 'Name'], axis=1, inplace=True, errors='ignore')
            # Mapping Placement(Y/N)? to status
            self.df.rename(columns={'Placement(Y/N)?': 'status'}, inplace=True)
            # Standardize status
            self.df['status'] = self.df['status'].map({'Placed': 'Placed', 'Not Placed': 'Not Placed'})
            
        return self.df

    def preprocess_data(self, target_col='status'):
        """Encoded categorical variables and splits features/target."""
        if self.df is None:
            self.load_data()
            
        temp_df = self.df.copy()
        
        categorical_cols = temp_df.select_dtypes(include=['object']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
            
        # Specific handling for status to ensure Placed=1, Not Placed=0
        temp_df['status'] = temp_df['status'].map({'Placed': 1, 'Not Placed': 0})
        
        for col in categorical_cols:
            le = LabelEncoder()
            temp_df[col] = le.fit_transform(temp_df[col].astype(str))
            self.encoders[col] = le
            
        # Drop rows with NaN in target (if any)
        temp_df.dropna(subset=[target_col], inplace=True)
            
        X = temp_df.drop(['status', 'salary'], axis=1, errors='ignore')
        y = temp_df['status']
        
        return X, y, self.encoders
