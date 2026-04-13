from src.data_manager import DataManager
from src.model_trainer import ModelTrainer
import pandas as pd

# Train MBA
print("Training MBA Models...")
dm_mba = DataManager(mode='MBA')
df_mba = dm_mba.load_data()
X_mba, y_mba, enc_mba = dm_mba.preprocess_data()
trainer_mba = ModelTrainer(mode='MBA')
print("  - Classification...")
trainer_mba.train_classification(X_mba, y_mba)

# Regression for MBA (only placed)
placed_df = df_mba[df_mba['status'] == 'Placed'].copy()
# Re-preprocess for regression (categorical to numeric)
for col, le in enc_mba.items():
    if col in placed_df.columns:
        placed_df[col] = le.transform(placed_df[col].astype(str))
X_reg = placed_df.drop(['status', 'salary'], axis=1, errors='ignore')
y_reg = placed_df['salary']
print("  - Regression...")
trainer_mba.train_regression(X_reg, y_reg)
trainer_mba.save_models(X_mba.columns.tolist(), enc_mba)

# Train Engineering
print("\nTraining Engineering Models...")
dm_eng = DataManager(mode='Engineering')
df_eng = dm_eng.load_data()
X_eng, y_eng, enc_eng = dm_eng.preprocess_data()
trainer_eng = ModelTrainer(mode='Engineering')
print("  - Classification...")
trainer_eng.train_classification(X_eng, y_eng)
trainer_eng.save_models(X_eng.columns.tolist(), enc_eng)

print("\nAll models trained and saved successfully!")
