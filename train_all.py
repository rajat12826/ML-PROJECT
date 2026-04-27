from src.data_manager import DataManager
from src.model_trainer import ModelTrainer
import pandas as pd
import joblib

def train_domain(mode='MBA'):
    print(f"\n--- Training {mode} Models ---")
    dm = DataManager(mode=mode)
    df = dm.load_data()
    X, y, enc = dm.preprocess_data()
    trainer = ModelTrainer(mode=mode)
    
    # 4 models for classification
    class_models = ['Gradient Boosting', 'Random Forest', 'Logistic Regression', 'Extra Trees']
    # 4 models for regression
    reg_models = ['Random Forest', 'Gradient Boosting', 'Linear Regression', 'Ridge Regression']
    
    for i in range(4):
        c_name = class_models[i]
        r_name = reg_models[i]
        
        print(f"  Training Set {i+1}: {c_name} / {r_name}...")
        
        # Classification
        class_metrics = trainer.train_classification(X, y, model_type=c_name)
        
        # Regression (only for MBA or where data exists)
        reg_metrics = None
        if mode == 'MBA':
            placed_df = df[df['status'] == 'Placed'].copy()
            if len(placed_df) > 0:
                X_reg_raw = placed_df[X.columns]
                y_reg = placed_df['salary']
                X_reg = X_reg_raw.copy()
                for col, le in enc.items():
                    if col in X_reg.columns:
                        def safe_transform(val):
                            val_str = str(val) if pd.notnull(val) else 'Unknown'
                            try: return le.transform([val_str])[0]
                            except: return 0
                        X_reg[col] = X_reg[col].apply(safe_transform)
                reg_metrics = trainer.train_regression(X_reg, y_reg, model_type=r_name)
        
        # Save this specific model combination
        trainer.save_models(X.columns.tolist(), enc, class_metrics, reg_metrics, model_name=c_name)
        print(f"    {c_name} Accuracy: {class_metrics['accuracy']:.2f}")

def main():
    train_domain(mode='MBA')
    train_domain(mode='Engineering')
    print("\nAll model suites tuned and saved!")

if __name__ == "__main__":
    main()
