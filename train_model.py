"""
ML Model Training for Dispersion Trading - THREE-CLASS VERSION
===============================================================

This script trains and evaluates machine learning models to predict
the best action for dispersion trades:
    Class 0: SHORT_DISPERSION (correlation will decrease)
    Class 1: LONG_DISPERSION (correlation will increase)
    Class 2: NO_TRADE (no significant movement)

Models evaluated:
1. Logistic Regression (baseline)
2. Random Forest
3. Gradient Boosting

The best model is saved for use in the live signal generator.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Class labels
CLASS_NAMES = {0: 'SHORT_DISPERSION', 1: 'LONG_DISPERSION', 2: 'NO_TRADE'}


class DispersionMLTrainer:
    """
    Trains and evaluates ML models for three-class dispersion trading.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the features dataset CSV
        """
        self.data_path = data_path or '/home/ubuntu/dispersion_live/ml_data/features_dataset.csv'
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
        # Feature columns
        self.feature_cols = [
            'z_score',
            'impl_corr_level',
            'vix_level',
            'vix_percentile',
            'vix_volatility',
            'vix_change_5d',
            'vix_change_20d',
            'corr_momentum_5d',
            'corr_momentum_20d',
            'corr_mean_reversion',
            'corr_percentile',
            'zscore_momentum_5d',
            'index_iv',
            'index_iv_change_5d',
            'vix_vxn_spread',
            'zscore_rolling_mean',
            'zscore_rolling_std',
            'days_since_extreme',
            'vix_regime'
        ]
        
        # Number of classes
        self.n_classes = 3
    
    def load_data(self):
        """Load the features dataset."""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(self.df)} rows")
        print(f"✓ Date range: {self.df.index.min().date()} to {self.df.index.max().date()}")
        
        # Drop rows with NaN in features or target
        required_cols = self.feature_cols + ['target']
        self.df = self.df.dropna(subset=required_cols)
        print(f"✓ After dropping NaN: {len(self.df)} rows")
        
        # Show class distribution
        print(f"\n✓ Class distribution:")
        for cls, name in CLASS_NAMES.items():
            count = (self.df['target'] == cls).sum()
            pct = count / len(self.df) * 100
            print(f"    {cls} ({name}): {count} ({pct:.1f}%)")
        
    def prepare_data(self, test_size: float = 0.2):
        """
        Prepare data for training.
        
        Uses time-based split to avoid look-ahead bias.
        """
        print("\n" + "=" * 60)
        print("PREPARING DATA")
        print("=" * 60)
        
        X = self.df[self.feature_cols].values
        y = self.df['target'].values.astype(int)
        
        # Time-based split (last 20% for testing)
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X[:split_idx]
        self.X_test = X[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✓ Training set: {len(self.X_train)} samples")
        print(f"✓ Test set: {len(self.X_test)} samples")
        print(f"✓ Training period: {self.df.index[0].date()} to {self.df.index[split_idx-1].date()}")
        print(f"✓ Test period: {self.df.index[split_idx].date()} to {self.df.index[-1].date()}")
        
        print(f"\n✓ Train class distribution:")
        for cls, name in CLASS_NAMES.items():
            count = (self.y_train == cls).sum()
            pct = count / len(self.y_train) * 100
            print(f"    {cls} ({name}): {count} ({pct:.1f}%)")
        
        print(f"\n✓ Test class distribution:")
        for cls, name in CLASS_NAMES.items():
            count = (self.y_test == cls).sum()
            pct = count / len(self.y_test) * 100
            print(f"    {cls} ({name}): {count} ({pct:.1f}%)")
        
    def train_models(self):
        """Train all models for three-class classification."""
        print("\n" + "=" * 60)
        print("TRAINING MODELS (THREE-CLASS)")
        print("=" * 60)
        
        # Model 1: Logistic Regression (multi-class)
        print("\n1. Logistic Regression (multinomial)...")
        lr = LogisticRegression(
            max_iter=1000, 
            random_state=42,
            solver='lbfgs'
        )
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['logistic_regression'] = lr
        print("   ✓ Trained")
        
        # Model 2: Random Forest
        print("\n2. Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)  # RF doesn't need scaling
        self.models['random_forest'] = rf
        print("   ✓ Trained")
        
        # Model 3: Gradient Boosting
        print("\n3. Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb.fit(self.X_train, self.y_train)
        self.models['gradient_boosting'] = gb
        print("   ✓ Trained")
        
    def evaluate_models(self):
        """Evaluate all models on test set."""
        print("\n" + "=" * 60)
        print("EVALUATING MODELS")
        print("=" * 60)
        
        for name, model in self.models.items():
            print(f"\n{name.upper()}")
            print("-" * 40)
            
            # Get predictions
            if name == 'logistic_regression':
                y_pred = model.predict(self.X_test_scaled)
                y_prob = model.predict_proba(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
                y_prob = model.predict_proba(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f} (weighted)")
            print(f"  Recall:    {recall:.4f} (weighted)")
            print(f"  F1 Score:  {f1:.4f} (weighted)")
            
            # Per-class accuracy
            print(f"\n  Per-class accuracy:")
            for cls, name_cls in CLASS_NAMES.items():
                mask = self.y_test == cls
                if mask.sum() > 0:
                    cls_acc = (y_pred[mask] == cls).mean()
                    print(f"    {name_cls}: {cls_acc:.2%}")
            
    def compare_to_baseline(self):
        """Compare ML models to the baseline Z-score strategy."""
        print("\n" + "=" * 60)
        print("COMPARISON TO BASELINE Z-SCORE STRATEGY")
        print("=" * 60)
        
        # Get test data with dates
        test_df = self.df.iloc[-len(self.y_test):].copy()
        
        # Baseline: Trade when |Z-score| > 1.5
        short_signals = test_df['z_score'] > 1.5
        long_signals = test_df['z_score'] < -1.5
        
        # SHORT baseline
        short_trades = test_df[short_signals]
        if len(short_trades) > 0:
            short_correct = (short_trades['target'] == 0).sum()
            short_win_rate = short_correct / len(short_trades)
        else:
            short_win_rate = 0
            short_correct = 0
        
        # LONG baseline
        long_trades = test_df[long_signals]
        if len(long_trades) > 0:
            long_correct = (long_trades['target'] == 1).sum()
            long_win_rate = long_correct / len(long_trades)
        else:
            long_win_rate = 0
            long_correct = 0
        
        print(f"\nBASELINE (Z-score threshold ±1.5):")
        print(f"  SHORT signals (Z > 1.5): {len(short_trades)} trades, {short_win_rate*100:.1f}% correct")
        print(f"  LONG signals (Z < -1.5): {len(long_trades)} trades, {long_win_rate*100:.1f}% correct")
        
        self.baseline_short_trades = len(short_trades)
        self.baseline_short_win_rate = short_win_rate
        self.baseline_long_trades = len(long_trades)
        self.baseline_long_win_rate = long_win_rate
        
        # ML models with probability threshold
        print(f"\nML MODELS (Probability > 0.50 for predicted class):")
        
        for name, result in self.results.items():
            y_prob = result['y_prob']
            y_pred = result['y_pred']
            
            # Only trade when max probability > threshold
            max_probs = y_prob.max(axis=1)
            confident_mask = max_probs > 0.50
            
            # SHORT trades (predicted class 0 with confidence)
            short_mask = (y_pred == 0) & confident_mask
            if short_mask.sum() > 0:
                ml_short_correct = (self.y_test[short_mask] == 0).sum()
                ml_short_win_rate = ml_short_correct / short_mask.sum()
            else:
                ml_short_correct = 0
                ml_short_win_rate = 0
            
            # LONG trades (predicted class 1 with confidence)
            long_mask = (y_pred == 1) & confident_mask
            if long_mask.sum() > 0:
                ml_long_correct = (self.y_test[long_mask] == 1).sum()
                ml_long_win_rate = ml_long_correct / long_mask.sum()
            else:
                ml_long_correct = 0
                ml_long_win_rate = 0
            
            print(f"\n  {name.upper()}:")
            print(f"    SHORT: {short_mask.sum()} trades, {ml_short_win_rate*100:.1f}% correct")
            print(f"    LONG: {long_mask.sum()} trades, {ml_long_win_rate*100:.1f}% correct")
            
            # Store for comparison
            self.results[name]['ml_short_trades'] = short_mask.sum()
            self.results[name]['ml_short_win_rate'] = ml_short_win_rate
            self.results[name]['ml_long_trades'] = long_mask.sum()
            self.results[name]['ml_long_win_rate'] = ml_long_win_rate
        
    def get_feature_importance(self):
        """Get feature importance from tree-based models."""
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE")
        print("=" * 60)
        
        # Use Random Forest for feature importance
        if 'random_forest' in self.models:
            model = self.models['random_forest']
            importances = model.feature_importances_
            
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            
            print("\nTop 10 Features (Random Forest):")
            for i in range(min(10, len(self.feature_cols))):
                idx = indices[i]
                print(f"  {i+1}. {self.feature_cols[idx]}: {importances[idx]:.4f}")
                
            return dict(zip(self.feature_cols, importances))
        
        return None
    
    def save_best_model(self, output_dir: str = None):
        """Save the best performing model."""
        if output_dir is None:
            output_dir = '/home/ubuntu/dispersion_live/ml_data'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find best model by F1 score (better for multi-class)
        best_name = max(self.results, key=lambda x: self.results[x]['f1'])
        best_model = self.models[best_name]
        
        print(f"\n✓ Best model: {best_name} (F1: {self.results[best_name]['f1']:.4f})")
        
        # Save model
        model_path = os.path.join(output_dir, 'best_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': best_model,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                'model_name': best_name,
                'n_classes': self.n_classes,
                'class_names': CLASS_NAMES,
                'metrics': self.results[best_name]
            }, f)
        
        print(f"✓ Model saved to: {model_path}")
        
        return model_path


def run_ml_training():
    """Main function to train and evaluate ML models."""
    print("=" * 70)
    print("ML MODEL TRAINING FOR DISPERSION TRADING (THREE-CLASS)")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Classes: SHORT_DISPERSION (0), LONG_DISPERSION (1), NO_TRADE (2)")
    print("=" * 70)
    
    # Initialize trainer
    trainer = DispersionMLTrainer()
    
    # Load data
    trainer.load_data()
    
    # Prepare data
    trainer.prepare_data(test_size=0.2)
    
    # Train models
    trainer.train_models()
    
    # Evaluate models
    trainer.evaluate_models()
    
    # Compare to baseline
    trainer.compare_to_baseline()
    
    # Get feature importance
    trainer.get_feature_importance()
    
    # Save best model
    trainer.save_best_model()
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────────────┐")
    print("│                       MODEL COMPARISON SUMMARY                           │")
    print("├─────────────────────┬──────────┬──────────────────┬──────────────────────┤")
    print("│ Model               │ F1 Score │ SHORT Win Rate   │ LONG Win Rate        │")
    print("├─────────────────────┼──────────┼──────────────────┼──────────────────────┤")
    print(f"│ Baseline (|Z|>1.5)  │   N/A    │ {trainer.baseline_short_win_rate*100:>6.1f}% ({trainer.baseline_short_trades:>3})   │ {trainer.baseline_long_win_rate*100:>6.1f}% ({trainer.baseline_long_trades:>3})        │")
    
    for name, result in trainer.results.items():
        short_trades = result.get('ml_short_trades', 0)
        short_wr = result.get('ml_short_win_rate', 0) * 100
        long_trades = result.get('ml_long_trades', 0)
        long_wr = result.get('ml_long_win_rate', 0) * 100
        print(f"│ {name:<19} │ {result['f1']:>8.4f} │ {short_wr:>6.1f}% ({short_trades:>3})   │ {long_wr:>6.1f}% ({long_trades:>3})        │")
    
    print("└─────────────────────┴──────────┴──────────────────┴──────────────────────┘")
    print("=" * 70)
    
    return trainer


if __name__ == "__main__":
    trainer = run_ml_training()
