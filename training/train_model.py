#!/usr/bin/env python3
"""
EarlyPark Model Training Pipeline
Downloads UCI Parkinson's dataset and trains models from scratch
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
import zipfile
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

class ParkinsonModelTrainer:
    """
    Comprehensive Parkinson's Disease Model Training Pipeline
    
    Supports:
    - Multiple datasets (UCI, custom CSV files)
    - Various ML algorithms
    - Hyperparameter tuning
    - Model evaluation and comparison
    - Feature importance analysis
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.best_model = None
        self.best_scaler = None
        self.feature_names = []
        
    def download_uci_dataset(self):
        """Download UCI Parkinson's dataset from the web"""
        print("📥 Downloading UCI Parkinson's Telemonitoring Dataset...")
        
        # Direct download URL for the dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Save the dataset
            with open('parkinsons_updrs.data', 'wb') as f:
                f.write(response.content)
            
            print("✅ Dataset downloaded successfully!")
            return 'parkinsons_updrs.data'
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("📖 Please manually download from: https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring")
            return None
    
    def load_uci_dataset(self, zip_path='parkinsons.zip'):
        """Load the UCI Parkinson's dataset (from zip or direct download)"""
        try:
            # First try to load from zip file (if provided)
            if os.path.exists(zip_path):
                nested_file_path = 'telemonitoring/parkinsons_updrs.data'
                
                with zipfile.ZipFile(zip_path, 'r') as z:
                    with z.open(nested_file_path) as f:
                        df = pd.read_csv(f)
                
                print(f"✅ UCI Dataset loaded from zip: {df.shape}")
                return self._preprocess_uci_data(df)
            
            # Try to load from direct file
            elif os.path.exists('parkinsons_updrs.data'):
                df = pd.read_csv('parkinsons_updrs.data')
                print(f"✅ UCI Dataset loaded from file: {df.shape}")
                return self._preprocess_uci_data(df)
            
            # Download the dataset
            else:
                dataset_file = self.download_uci_dataset()
                if dataset_file:
                    df = pd.read_csv(dataset_file)
                    print(f"✅ UCI Dataset loaded after download: {df.shape}")
                    return self._preprocess_uci_data(df)
                else:
                    return None
            
        except Exception as e:
            print(f"❌ Failed to load UCI dataset: {e}")
            return None
    
    def _preprocess_uci_data(self, df):
        """Preprocess UCI Parkinson's dataset"""
        # Remove duplicates and identifier columns
        df = df.drop_duplicates()
        df = df.drop(columns=['subject#'], errors='ignore')
        
        # Define target and features
        target_column = 'total_UPDRS'
        
        # Remove motor_UPDRS as it's highly correlated with total_UPDRS
        features = df.drop(columns=[target_column, 'motor_UPDRS'], errors='ignore')
        target = df[target_column]
        
        return features, target
    
    def load_custom_dataset(self, csv_path, target_column='total_UPDRS'):
        """Load a custom CSV dataset"""
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Custom dataset loaded: {df.shape}")
            
            # Basic preprocessing
            df = df.dropna()
            df = df.drop_duplicates()
            
            # Remove identifier columns if present
            id_columns = ['subject#', 'subject_id', 'patient_id', 'id']
            for col in id_columns:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            return self._prepare_features_target(df, target_column)
            
        except Exception as e:
            print(f"❌ Failed to load custom dataset: {e}")
            return None
    
    def _preprocess_uci_data(self, df):
        """Preprocess UCI Parkinson's dataset"""
        # Remove duplicates and identifier columns
        df = df.drop_duplicates()
        df = df.drop(columns=['subject#'], errors='ignore')
        
        # Define target and features
        target_column = 'total_UPDRS'
        
        # Remove motor_UPDRS as it's highly correlated with total_UPDRS
        features = df.drop(columns=[target_column, 'motor_UPDRS'], errors='ignore')
        target = df[target_column]
        
        return features, target
    
    def _prepare_features_target(self, df, target_column):
        """Prepare features and target from DataFrame"""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        features = df.drop(columns=[target_column])
        target = df[target_column]
        
        return features, target
    
    def train_multiple_models(self, X, y, test_size=0.2, random_state=42):
        """Train and compare multiple ML models"""
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to try
        models_config = {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 4]
                }
            },
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 1],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
        
        print("🚀 Training multiple models...")
        print("=" * 50)
        
        best_score = float('inf')
        
        for name, config in models_config.items():
            print(f"\n📊 Training {name}...")
            
            # Hyperparameter tuning if params provided
            if config['params']:
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'],
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
                print(f"Best params: {grid_search.best_params_}")
            else:
                best_model = config['model']
                best_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = best_model.predict(X_train_scaled)
            y_pred_test = best_model.predict(X_test_scaled)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                best_model, X_train_scaled, y_train, 
                cv=5, scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Store results
            self.models[name] = best_model
            self.scalers[name] = scaler
            self.results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'cv_rmse': cv_rmse,
                'predictions': y_pred_test,
                'test_targets': y_test
            }
            
            print(f"Train RMSE: {train_rmse:.3f}")
            print(f"Test RMSE: {test_rmse:.3f}")
            print(f"Test MAE: {test_mae:.3f}")
            print(f"Test R²: {test_r2:.3f}")
            print(f"CV RMSE: {cv_rmse:.3f}")
            
            # Track best model
            if test_rmse < best_score:
                best_score = test_rmse
                self.best_model = best_model
                self.best_scaler = scaler
                print(f"🏆 New best model: {name}")
        
        return self.results
    
    def plot_model_comparison(self, save_path=None):
        """Plot model comparison results"""
        if not self.results:
            print("❌ No results to plot. Train models first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.results.keys())
        
        # RMSE Comparison
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        cv_rmse = [self.results[m]['cv_rmse'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0,0].bar(x - width/2, test_rmse, width, label='Test RMSE')
        axes[0,0].bar(x + width/2, cv_rmse, width, label='CV RMSE')
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].set_title('RMSE Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(models, rotation=45)
        axes[0,0].legend()
        
        # R² Comparison
        r2_scores = [self.results[m]['test_r2'] for m in models]
        axes[0,1].bar(models, r2_scores, color='green', alpha=0.7)
        axes[0,1].set_xlabel('Models')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].set_title('R² Score Comparison')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Prediction vs Actual (Best Model)
        best_model_name = min(models, key=lambda m: self.results[m]['test_rmse'])
        best_result = self.results[best_model_name]
        
        axes[1,0].scatter(best_result['test_targets'], best_result['predictions'], alpha=0.6)
        axes[1,0].plot([min(best_result['test_targets']), max(best_result['test_targets'])],
                       [min(best_result['test_targets']), max(best_result['test_targets'])],
                       'r--', lw=2)
        axes[1,0].set_xlabel('Actual UPDRS')
        axes[1,0].set_ylabel('Predicted UPDRS')
        axes[1,0].set_title(f'Prediction vs Actual ({best_model_name})')
        
        # Residuals plot
        residuals = best_result['test_targets'] - best_result['predictions']
        axes[1,1].scatter(best_result['predictions'], residuals, alpha=0.6)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_xlabel('Predicted UPDRS')
        axes[1,1].set_ylabel('Residuals')
        axes[1,1].set_title(f'Residuals Plot ({best_model_name})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
    
    def analyze_feature_importance(self, model_name=None):
        """Analyze feature importance for tree-based models"""
        if model_name is None:
            # Use best model
            model_name = min(self.results.keys(), key=lambda m: self.results[m]['test_rmse'])
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🎯 Feature Importance ({model_name}):")
            print("=" * 40)
            print(importance_df.head(10).to_string(index=False))
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(15), y='feature', x='importance')
            plt.title(f'Top 15 Feature Importance ({model_name})')
            plt.tight_layout()
            plt.show()
            
            return importance_df
        else:
            print(f"❌ Model {model_name} doesn't support feature importance")
            return None
    
    def save_best_model(self, model_path='final_parkinsons_regressor.pkl', 
                       scaler_path='parkinsons_scaler.pkl'):
        """Save the best performing model and scaler"""
        if self.best_model is None:
            print("❌ No trained model to save. Train models first.")
            return False
        
        try:
            # Save model and scaler
            joblib.dump(self.best_model, model_path)
            joblib.dump(self.best_scaler, scaler_path)
            
            print(f"✅ Best model saved to: {model_path}")
            print(f"✅ Scaler saved to: {scaler_path}")
            
            # Print model summary
            best_model_name = min(self.results.keys(), key=lambda m: self.results[m]['test_rmse'])
            best_result = self.results[best_model_name]
            
            print(f"\n📊 Best Model Summary ({best_model_name}):")
            print("=" * 40)
            print(f"Test RMSE: {best_result['test_rmse']:.3f}")
            print(f"Test R²: {best_result['test_r2']:.3f}")
            print(f"Test MAE: {best_result['test_mae']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False

# Example usage and training script
def main():
    """Main training pipeline"""
    print("🧠 Parkinson's Disease Model Training Pipeline")
    print("=" * 60)
    
    trainer = ParkinsonModelTrainer()
    
    # Load dataset
    print("\n📂 Loading dataset...")
    
    # Try UCI dataset first
    data = trainer.load_uci_dataset('parkinsons.zip')
    
    if data is None:
        print("⚠️ UCI dataset not found. Please provide dataset path.")
        # Example: Load custom CSV
        # data = trainer.load_custom_dataset('your_dataset.csv', target_column='total_UPDRS')
        return
    
    X, y = data
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")
    
    # Train models
    print("\n🏋️ Training models...")
    results = trainer.train_multiple_models(X, y)
    
    # Plot comparisons
    print("\n📈 Generating plots...")
    trainer.plot_model_comparison('model_comparison.png')
    
    # Feature importance
    trainer.analyze_feature_importance()
    
    # Save best model
    print("\n💾 Saving best model...")
    trainer.save_best_model()
    
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()
    