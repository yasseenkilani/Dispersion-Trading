"""
Model Validation Module for Dispersion Trading System

Implements:
- Purged K-Fold Cross-Validation (prevents data leakage)
- Feature Importance Analysis
- Hyperparameter Optimization
- Model Performance Metrics

Key Concept: Purged K-Fold
- Standard K-fold can cause data leakage in time series
- Purged K-fold adds a gap between train/test sets
- This prevents look-ahead bias from autocorrelation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import joblib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURATION
# =============================================================================

ML_DATA_DIR = "ml_data"
FEATURES_FILE = os.path.join(ML_DATA_DIR, "features_dataset.csv")
PURGE_GAP_DAYS = 20  # Days to exclude between train/test to prevent leakage
N_FOLDS = 5  # Number of folds for cross-validation


# =============================================================================
# PURGED K-FOLD CROSS-VALIDATION
# =============================================================================

class PurgedKFold:
    """
    Purged K-Fold Cross-Validation for Time Series Data.
    
    Unlike standard K-fold, this implementation:
    1. Respects temporal ordering (no shuffling)
    2. Adds a gap between train and test sets to prevent leakage
    3. Prevents information from the test period leaking into training
    
    The purge gap accounts for autocorrelation in financial time series.
    """
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 20):
        """
        Initialize PurgedKFold.
        
        Args:
            n_splits: Number of folds
            purge_gap: Number of samples to exclude between train and test
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups=None):
        """
        Generate indices to split data into training and test sets.
        
        Yields:
            train_idx: Training set indices
            test_idx: Test set indices
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Test set is the i-th fold
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            test_idx = np.arange(test_start, test_end)
            
            # Training set is everything else, with purge gap
            train_idx = []
            
            # Add samples before test set (with purge gap)
            if test_start > 0:
                train_end = max(0, test_start - self.purge_gap)
                train_idx.extend(range(0, train_end))
            
            # Add samples after test set (with purge gap)
            if test_end < n_samples:
                train_start = min(n_samples, test_end + self.purge_gap)
                train_idx.extend(range(train_start, n_samples))
            
            if len(train_idx) > 0:
                yield np.array(train_idx), test_idx
    
    def get_n_splits(self):
        return self.n_splits


class WalkForwardPurgedCV:
    """
    Walk-Forward Cross-Validation with Purging.
    
    More appropriate for trading systems as it:
    1. Only trains on past data
    2. Tests on future data
    3. Adds purge gap to prevent leakage
    """
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 20, 
                 min_train_size: float = 0.3):
        """
        Initialize WalkForwardPurgedCV.
        
        Args:
            n_splits: Number of test periods
            purge_gap: Number of samples to exclude between train and test
            min_train_size: Minimum training set size as fraction of data
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.min_train_size = min_train_size
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups=None):
        """
        Generate indices for walk-forward validation.
        
        Yields:
            train_idx: Training set indices (past data only)
            test_idx: Test set indices (future data)
        """
        n_samples = len(X)
        min_train = int(n_samples * self.min_train_size)
        
        # Calculate test fold size
        remaining = n_samples - min_train
        fold_size = remaining // self.n_splits
        
        for i in range(self.n_splits):
            # Test set
            test_start = min_train + i * fold_size
            test_end = min_train + (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            test_idx = np.arange(test_start, test_end)
            
            # Training set is all data before test set, minus purge gap
            train_end = max(0, test_start - self.purge_gap)
            train_idx = np.arange(0, train_end)
            
            if len(train_idx) > 0:
                yield train_idx, test_idx
    
    def get_n_splits(self):
        return self.n_splits


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance using multiple methods.
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize analyzer.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        self.importance_results = {}
    
    def analyze_logistic_regression(self, model, X: np.ndarray) -> pd.DataFrame:
        """
        Get feature importance from Logistic Regression coefficients.
        
        Args:
            model: Trained LogisticRegression model
            X: Feature matrix (for scaling reference)
            
        Returns:
            DataFrame with feature importances
        """
        # Get absolute coefficients
        coefficients = np.abs(model.coef_[0])
        
        # Normalize to sum to 1
        importance = coefficients / coefficients.sum()
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'coefficient': model.coef_[0]
        }).sort_values('importance', ascending=False)
        
        self.importance_results['logistic_regression'] = df
        return df
    
    def analyze_random_forest(self, model) -> pd.DataFrame:
        """
        Get feature importance from Random Forest.
        
        Args:
            model: Trained RandomForestClassifier
            
        Returns:
            DataFrame with feature importances
        """
        importance = model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.importance_results['random_forest'] = df
        return df
    
    def analyze_permutation_importance(self, model, X: np.ndarray, y: np.ndarray,
                                       n_repeats: int = 10) -> pd.DataFrame:
        """
        Calculate permutation importance.
        
        Measures how much model performance drops when each feature is shuffled.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            n_repeats: Number of times to shuffle each feature
            
        Returns:
            DataFrame with permutation importances
        """
        baseline_score = accuracy_score(y, model.predict(X))
        importances = []
        
        for i, feature in enumerate(self.feature_names):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                score = accuracy_score(y, model.predict(X_permuted))
                scores.append(baseline_score - score)
            
            importances.append(np.mean(scores))
        
        # Normalize
        importances = np.array(importances)
        importances = np.maximum(importances, 0)  # No negative importance
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.importance_results['permutation'] = df
        return df
    
    def get_combined_importance(self) -> pd.DataFrame:
        """
        Combine importance from all methods.
        
        Returns:
            DataFrame with combined feature importances
        """
        if not self.importance_results:
            raise ValueError("No importance analysis has been run yet")
        
        # Start with feature names
        combined = pd.DataFrame({'feature': self.feature_names})
        
        # Add each method's importance
        for method, df in self.importance_results.items():
            combined = combined.merge(
                df[['feature', 'importance']].rename(columns={'importance': method}),
                on='feature'
            )
        
        # Calculate average importance
        importance_cols = [c for c in combined.columns if c != 'feature']
        combined['avg_importance'] = combined[importance_cols].mean(axis=1)
        
        return combined.sort_values('avg_importance', ascending=False)
    
    def plot_importance(self, save_path: str = None):
        """
        Plot feature importance comparison.
        
        Args:
            save_path: Path to save the plot
        """
        combined = self.get_combined_importance()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # Bar chart of average importance
        ax1 = axes[0]
        top_features = combined.head(15)
        ax1.barh(range(len(top_features)), top_features['avg_importance'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('Average Importance')
        ax1.set_title('Top 15 Features by Average Importance')
        ax1.invert_yaxis()
        
        # Comparison across methods
        ax2 = axes[1]
        methods = [c for c in combined.columns if c not in ['feature', 'avg_importance']]
        x = np.arange(len(top_features))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            ax2.barh(x + i * width, top_features[method], width, label=method)
        
        ax2.set_yticks(x + width * len(methods) / 2)
        ax2.set_yticklabels(top_features['feature'])
        ax2.set_xlabel('Importance')
        ax2.set_title('Feature Importance by Method')
        ax2.legend()
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.close()
        return combined


# =============================================================================
# MODEL VALIDATOR
# =============================================================================

class ModelValidator:
    """
    Comprehensive model validation with purged cross-validation.
    """
    
    def __init__(self, features_file: str = FEATURES_FILE):
        """
        Initialize validator.
        
        Args:
            features_file: Path to features dataset
        """
        self.features_file = features_file
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.results = {}
    
    def load_data(self):
        """Load and prepare the features dataset."""
        print("Loading features dataset...")
        self.df = pd.read_csv(self.features_file)
        
        # Sort by date to ensure temporal ordering
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Separate features and target
        exclude_cols = ['date', 'target', 'target_short', 'target_long', 'target_pnl',
                        'future_return', 'future_corr', 'corr_change', 'short_pnl', 'long_pnl',
                        'impl_corr', 'hist_corr']
        self.feature_names = [c for c in self.df.columns if c not in exclude_cols]
        
        # Drop rows with NaN values in features or target
        print(f"  Original samples: {len(self.df)}")
        
        # Check for NaN in target
        valid_target = ~self.df['target'].isna()
        
        # Check for NaN in features
        valid_features = ~self.df[self.feature_names].isna().any(axis=1)
        
        # Keep only valid rows
        valid_mask = valid_target & valid_features
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        print(f"  After dropping NaN: {len(self.df)} samples")
        
        self.X = self.df[self.feature_names].values
        
        # Convert target to binary: 1 = SHORT wins (target 1 or 2), 0 = LONG wins (target 0)
        # Original: 0=LONG wins, 1=SHORT wins, 2=both win
        self.y = (self.df['target'] >= 1).astype(int).values
        
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Target distribution: {np.mean(self.y):.1%} positive (SHORT wins)")
        print(f"  Feature names: {self.feature_names[:5]}... (showing first 5)")
        
        return self
    
    def run_purged_cv(self, model_class, model_params: Dict = None,
                      cv_type: str = 'walk_forward') -> Dict:
        """
        Run purged cross-validation.
        
        Args:
            model_class: Model class (e.g., LogisticRegression)
            model_params: Model hyperparameters
            cv_type: 'purged' or 'walk_forward'
            
        Returns:
            Dict with CV results
        """
        if model_params is None:
            model_params = {}
        
        # Select CV strategy
        if cv_type == 'purged':
            cv = PurgedKFold(n_splits=N_FOLDS, purge_gap=PURGE_GAP_DAYS)
        else:
            cv = WalkForwardPurgedCV(n_splits=N_FOLDS, purge_gap=PURGE_GAP_DAYS)
        
        print(f"\nRunning {cv_type} cross-validation...")
        print(f"  Folds: {N_FOLDS}")
        print(f"  Purge gap: {PURGE_GAP_DAYS} days")
        
        # Store fold results
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y)):
            # Split data
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = {
                'fold': fold + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
            }
            
            fold_results.append(metrics)
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)
            
            print(f"  Fold {fold + 1}: Acc={metrics['accuracy']:.3f}, AUC={metrics['auc']:.3f}")
        
        # Aggregate results
        results_df = pd.DataFrame(fold_results)
        
        summary = {
            'cv_type': cv_type,
            'n_folds': N_FOLDS,
            'purge_gap': PURGE_GAP_DAYS,
            'accuracy_mean': results_df['accuracy'].mean(),
            'accuracy_std': results_df['accuracy'].std(),
            'precision_mean': results_df['precision'].mean(),
            'recall_mean': results_df['recall'].mean(),
            'f1_mean': results_df['f1'].mean(),
            'auc_mean': results_df['auc'].mean(),
            'auc_std': results_df['auc'].std(),
            'fold_results': results_df
        }
        
        print(f"\n  Summary:")
        print(f"    Accuracy: {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f}")
        print(f"    AUC: {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}")
        print(f"    Precision: {summary['precision_mean']:.3f}")
        print(f"    Recall: {summary['recall_mean']:.3f}")
        
        return summary
    
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyze feature importance using multiple methods.
        
        Returns:
            DataFrame with feature importances
        """
        print("\nAnalyzing feature importance...")
        
        # Scale all data
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Initialize analyzer
        analyzer = FeatureImportanceAnalyzer(self.feature_names)
        
        # Train Logistic Regression
        lr_model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight='balanced'
        )
        lr_model.fit(X_scaled, self.y)
        analyzer.analyze_logistic_regression(lr_model, X_scaled)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'
        )
        rf_model.fit(X_scaled, self.y)
        analyzer.analyze_random_forest(rf_model)
        
        # Permutation importance
        analyzer.analyze_permutation_importance(lr_model, X_scaled, self.y)
        
        # Get combined importance
        combined = analyzer.get_combined_importance()
        
        # Save plot
        plot_path = os.path.join(ML_DATA_DIR, 'feature_importance.png')
        analyzer.plot_importance(plot_path)
        
        # Print results
        print("\n  Top 10 Features:")
        print("  " + "-" * 60)
        for i, row in combined.head(10).iterrows():
            print(f"  {row['feature']:<30} {row['avg_importance']:.3f}")
        
        # Identify features to potentially prune
        low_importance = combined[combined['avg_importance'] < 0.03]
        if len(low_importance) > 0:
            print(f"\n  Features with <3% importance (consider pruning):")
            for _, row in low_importance.iterrows():
                print(f"    {row['feature']}: {row['avg_importance']:.3f}")
        
        self.results['feature_importance'] = combined
        return combined
    
    def optimize_hyperparameters(self, model_class, param_grid: Dict) -> Dict:
        """
        Optimize hyperparameters using purged cross-validation.
        
        Args:
            model_class: Model class
            param_grid: Dict of parameter ranges
            
        Returns:
            Dict with best parameters and results
        """
        print("\nOptimizing hyperparameters...")
        
        cv = WalkForwardPurgedCV(n_splits=N_FOLDS, purge_gap=PURGE_GAP_DAYS)
        
        best_score = 0
        best_params = None
        all_results = []
        
        param_combinations = list(ParameterGrid(param_grid))
        print(f"  Testing {len(param_combinations)} parameter combinations...")
        
        for params in param_combinations:
            scores = []
            
            for train_idx, test_idx in cv.split(self.X, self.y):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                model = model_class(**params)
                model.fit(X_train_scaled, y_train)
                
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
                scores.append(auc)
            
            mean_score = np.mean(scores)
            all_results.append({**params, 'mean_auc': mean_score, 'std_auc': np.std(scores)})
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        results_df = pd.DataFrame(all_results).sort_values('mean_auc', ascending=False)
        
        print(f"\n  Best Parameters:")
        for k, v in best_params.items():
            print(f"    {k}: {v}")
        print(f"  Best AUC: {best_score:.4f}")
        
        self.results['hyperparameter_optimization'] = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results_df
        }
        
        return self.results['hyperparameter_optimization']
    
    def run_full_validation(self) -> Dict:
        """
        Run complete validation pipeline.
        
        Returns:
            Dict with all validation results
        """
        print("=" * 70)
        print("MODEL VALIDATION PIPELINE")
        print("=" * 70)
        
        # Load data
        self.load_data()
        
        # Run purged cross-validation
        self.results['purged_cv'] = self.run_purged_cv(
            LogisticRegression,
            {'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced'},
            cv_type='purged'
        )
        
        self.results['walk_forward_cv'] = self.run_purged_cv(
            LogisticRegression,
            {'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced'},
            cv_type='walk_forward'
        )
        
        # Feature importance
        self.analyze_feature_importance()
        
        # Hyperparameter optimization
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'class_weight': ['balanced', None],
            'max_iter': [1000]
        }
        self.optimize_hyperparameters(LogisticRegression, param_grid)
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        print(f"\n  Purged K-Fold CV:")
        print(f"    Accuracy: {self.results['purged_cv']['accuracy_mean']:.3f} ± {self.results['purged_cv']['accuracy_std']:.3f}")
        print(f"    AUC: {self.results['purged_cv']['auc_mean']:.3f} ± {self.results['purged_cv']['auc_std']:.3f}")
        
        print(f"\n  Walk-Forward CV:")
        print(f"    Accuracy: {self.results['walk_forward_cv']['accuracy_mean']:.3f} ± {self.results['walk_forward_cv']['accuracy_std']:.3f}")
        print(f"    AUC: {self.results['walk_forward_cv']['auc_mean']:.3f} ± {self.results['walk_forward_cv']['auc_std']:.3f}")
        
        print(f"\n  Best Hyperparameters:")
        for k, v in self.results['hyperparameter_optimization']['best_params'].items():
            print(f"    {k}: {v}")
        
        # Save results
        results_file = os.path.join(ML_DATA_DIR, 'validation_results.json')
        save_results = {
            'purged_cv': {k: v for k, v in self.results['purged_cv'].items() if k != 'fold_results'},
            'walk_forward_cv': {k: v for k, v in self.results['walk_forward_cv'].items() if k != 'fold_results'},
            'best_params': self.results['hyperparameter_optimization']['best_params'],
            'best_auc': self.results['hyperparameter_optimization']['best_score'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_file, 'w') as f:
            import json
            json.dump(save_results, f, indent=2)
        
        print(f"\n  Results saved to: {results_file}")
        print("=" * 70)
        
        return self.results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the validation pipeline."""
    validator = ModelValidator()
    results = validator.run_full_validation()
    
    return results


if __name__ == "__main__":
    main()
