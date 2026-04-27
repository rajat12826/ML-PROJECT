from src.data_manager import DataManager
from src.model_trainer import ModelTrainer
import pandas as pd
import joblib

def main():
    # 1. MBA Training
    print("Training Balanced MBA Models...")
    dm_mba = DataManager(mode='MBA')
    df_mba = dm_mba.load_data()
    X_mba, y_mba, enc_mba = dm_mba.preprocess_data()
    trainer_mba = ModelTrainer(mode='MBA')
    
    class_metrics = trainer_mba.train_classification(X_mba, y_mba)
    print(f"  MBA Class Accuracy: {class_metrics['accuracy']:.2f}, F1: {class_metrics['f1_score']:.2f}")

    # Prepare for Salary regression - safe encoding
    placed_df = df_mba[df_mba['status'] == 'Placed'].copy()
    
    # We must ensure we only use columns that the classifier used
    X_reg_raw = placed_df[X_mba.columns]
    y_reg = placed_df['salary']
    
    # Safe categorical encoding for regression
    X_reg = X_reg_raw.copy()
    for col, le in enc_mba.items():
        if col in X_reg.columns:
            # Handle unseen 'nan' or other labels by ignoring or mapping to first class
            def safe_transform(val):
                val_str = str(val) if pd.notnull(val) else 'Unknown'
                try:
                    return le.transform([val_str])[0]
                except:
                    return 0 # Fallback
            X_reg[col] = X_reg[col].apply(safe_transform)
    
    reg_metrics = trainer_mba.train_regression(X_reg, y_reg)
    if reg_metrics:
        print(f"  MBA Reg RMSE: {reg_metrics['rmse']:.2f}")

    all_metrics_mba = {"classification": class_metrics, "regression": reg_metrics}
    trainer_mba.save_models(X_mba.columns.tolist(), enc_mba, metrics=all_metrics_mba)

    # 2. Engineering Training
    print("\nTraining Cleaned Engineering Models...")
    dm_eng = DataManager(mode='Engineering')
    df_eng = dm_eng.load_data()
    X_eng, y_eng, enc_eng = dm_eng.preprocess_data()
    trainer_eng = ModelTrainer(mode='Engineering')
    
    eng_metrics = trainer_eng.train_classification(X_eng, y_eng)
    print(f"  Eng Class Accuracy: {eng_metrics['accuracy']:.2f}, F1: {eng_metrics['f1_score']:.2f}")

    all_metrics_eng = {"classification": eng_metrics, "regression": None}
    trainer_eng.save_models(X_eng.columns.tolist(), enc_eng, metrics=all_metrics_eng)

    print("\nAll models tuned and saved with detailed metrics!")

if __name__ == "__main__":
    main()
