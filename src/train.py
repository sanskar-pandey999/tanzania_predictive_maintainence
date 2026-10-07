import os
import sys

# Ensure current script folder is on system path for module imports
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from preprocess import preprocess_df

def main():
    data_dir = os.path.abspath(os.path.join(src_dir, "..", "data"))
    
    val_file = os.path.join(data_dir, "independent.csv")
    label_file = os.path.join(data_dir, "dependent.csv")
    pred_file = os.path.join(data_dir, "prediction.csv")
    
    print(f"Loading training features: {os.path.basename(val_file)}")
    print(f"Loading target labels: {os.path.basename(label_file)}")
    print(f"Target prediction output: {os.path.basename(pred_file)}")
    
    # 1. Load Datasets
    df_val = pd.read_csv(val_file)
    df_label = pd.read_csv(label_file)
    df_pred = pd.read_csv(pred_file)
    
    # Merge training features and target labels on 'id'
    df_train = pd.merge(df_val, df_label[['id', 'status_group']], on='id')
        
    # Fit target label encoder
    target_encoder = LabelEncoder()
    df_train['target'] = target_encoder.fit_transform(df_train['status_group'])
    
    y_train_full = df_train['target']
    X_train_raw = df_train.drop(columns=['id', 'status_group', 'target'], errors='ignore')
    
    # 2. Preprocess Training Features
    X_train_proc, encoders = preprocess_df(X_train_raw, is_train=True)
    
    # Stratified Train/Validation Split (80/20)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_proc, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print("\n=======================================================")
    print("STEP 1: BASELINE MODEL (WITHOUT GRID SEARCH CV)")
    print("=======================================================")
    rf_baseline = RandomForestClassifier(
        n_estimators=15, max_depth=8, max_features=5, random_state=42
    )
    rf_baseline.fit(X_tr, y_tr)
    rf_base_pred = rf_baseline.predict(X_val)
    rf_base_acc = accuracy_score(y_val, rf_base_pred)
    print(f"Baseline Accuracy (Without Grid Search): {rf_base_acc * 100:.2f}%")
    
    print("\n=======================================================")
    print("STEP 2: HYPERPARAMETER TUNING VIA GRID SEARCH CV")
    print("=======================================================")
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [20, 25],
        'min_samples_split': [5, 10]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=rf_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    rf_grid.fit(X_tr, y_tr)
    
    best_model = rf_grid.best_estimator_
    rf_tuned_pred = best_model.predict(X_val)
    rf_tuned_acc = accuracy_score(y_val, rf_tuned_pred)
    
    print(f"\nOptimal Hyperparameters Found: {rf_grid.best_params_}")
    print(f"Tuned Random Forest Accuracy (With Grid Search): {rf_tuned_acc * 100:.2f}%")
    print(f"Accuracy Gain from Grid Search: +{(rf_tuned_acc - rf_base_acc) * 100:.2f}%")
    
    print("\n--- Validation Classification Metrics ---")
    print(classification_report(y_val, rf_tuned_pred, target_names=target_encoder.classes_))
    
    # 3. Predict on Test/Prediction Data
    print("\n=======================================================")
    print("STEP 3: PREDICTING AND UPDATING prediction.csv")
    print("=======================================================")
    
    # Retrain tuned model on full training set
    best_model.fit(X_train_proc, y_train_full)
    
    # Preprocess prediction features
    X_test_raw = df_pred.drop(columns=['id', 'status_group'], errors='ignore')
    X_test_proc, _ = preprocess_df(X_test_raw, is_train=False, encoders=encoders)
    
    # Align feature columns with training set
    for col in X_train_proc.columns:
        if col not in X_test_proc.columns:
            X_test_proc[col] = 0
    X_test_proc = X_test_proc[X_train_proc.columns]
    
    # Predict and map numeric classes back to original labels
    preds_num = best_model.predict(X_test_proc)
    preds_labels = target_encoder.inverse_transform(preds_num)
    
    # Export id and status_group into prediction.csv
    df_output = pd.DataFrame({
        'id': df_pred['id'],
        'status_group': preds_labels
    })
    
    df_output.to_csv(pred_file, index=False)
    
    print(f"Successfully updated predictions in: {pred_file}")
    print(f"\nFirst 10 predictions written:\n")
    print(df_output.head(10))

if __name__ == "__main__":
    main()