"""
Feature Engineer Module for ML-Enhanced Dispersion Trading
===========================================================

This module creates a rich feature set for machine learning model training.
Instead of relying solely on the Z-score threshold, the ML model will learn
which market conditions lead to profitable dispersion trades.

Features:
1. Core Signal: Implied Correlation Z-score
2. Market Regime: VIX level, VIX percentile
3. Volatility-of-Volatility: 20-day std dev of VIX
4. Momentum: 5-day and 20-day momentum of implied correlation
5. Mean Reversion: Distance from 252-day moving average
6. Term Structure: VIX vs VXN spread (when available)

Target Variable:
- y = 1 if a SHORT_DISPERSION trade entered on this day would be profitable
- y = 0 otherwise
- Profitability is measured after 5 trading days (holding period)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Creates ML features from historical dispersion trading data.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the feature engineer.
        
        Args:
            data_dir: Directory containing historical data files
        """
        self.data_dir = data_dir or '/home/ubuntu/qqq_backtest/data'
        self.correlation_file = '/home/ubuntu/dispersion_live/historical_data/correlation_data.csv'
        
        # Data containers
        self.correlation_data = None
        self.vix_data = None
        self.qqq_iv_data = None
        self.features_df = None
        
    def load_data(self):
        """Load all required historical data."""
        print("=" * 60)
        print("LOADING HISTORICAL DATA")
        print("=" * 60)
        
        # Load correlation data (has impl_corr, z_score, etc.)
        self.correlation_data = pd.read_csv(self.correlation_file)
        self.correlation_data['date'] = pd.to_datetime(self.correlation_data['date'])
        self.correlation_data = self.correlation_data.set_index('date').sort_index()
        print(f"✓ Correlation data: {len(self.correlation_data)} rows")
        print(f"  Date range: {self.correlation_data.index.min().date()} to {self.correlation_data.index.max().date()}")
        
        # Load VIX data
        vix_file = os.path.join(self.data_dir, 'vix_data.csv')
        self.vix_data = pd.read_csv(vix_file)
        self.vix_data['date'] = pd.to_datetime(self.vix_data['date'])
        self.vix_data = self.vix_data.set_index('date').sort_index()
        print(f"✓ VIX data: {len(self.vix_data)} rows")
        
        # Load QQQ IV data
        qqq_iv_file = os.path.join(self.data_dir, 'qqq_iv_data.csv')
        self.qqq_iv_data = pd.read_csv(qqq_iv_file)
        self.qqq_iv_data['date'] = pd.to_datetime(self.qqq_iv_data['date'])
        self.qqq_iv_data = self.qqq_iv_data.set_index('date').sort_index()
        print(f"✓ QQQ IV data: {len(self.qqq_iv_data)} rows")
        
        # Try to load VXN data if available
        vxn_file = os.path.join(self.data_dir, 'vxn_data.csv')
        if os.path.exists(vxn_file):
            self.vxn_data = pd.read_csv(vxn_file)
            self.vxn_data['date'] = pd.to_datetime(self.vxn_data['date'])
            self.vxn_data = self.vxn_data.set_index('date').sort_index()
            print(f"✓ VXN data: {len(self.vxn_data)} rows")
        else:
            self.vxn_data = None
            print("⚠ VXN data not available")
        
        print("=" * 60)
        
    def create_features(self) -> pd.DataFrame:
        """
        Create the full feature set for each day.
        
        Returns:
            DataFrame with all features
        """
        print("\n" + "=" * 60)
        print("CREATING FEATURES")
        print("=" * 60)
        
        # Start with correlation data as base
        df = self.correlation_data.copy()
        
        # Merge VIX data
        df = df.join(self.vix_data, how='left')
        
        # Merge QQQ IV data
        df = df.join(self.qqq_iv_data, how='left', rsuffix='_qqq')
        
        # Merge VXN data if available
        if self.vxn_data is not None:
            df = df.join(self.vxn_data, how='left')
            df.rename(columns={'vxn': 'vxn_level'}, inplace=True)
        
        print(f"Base dataframe: {len(df)} rows")
        
        # =====================================================
        # FEATURE 1: Core Signal - Z-score (already exists)
        # =====================================================
        # z_score column already exists in correlation_data
        print("✓ Feature 1: Z-score (existing)")
        
        # =====================================================
        # FEATURE 2: Implied Correlation Level
        # =====================================================
        df['impl_corr_level'] = df['impl_corr']
        print("✓ Feature 2: Implied correlation level")
        
        # =====================================================
        # FEATURE 3: VIX Level (Market Regime)
        # =====================================================
        df['vix_level'] = df['vix']
        print("✓ Feature 3: VIX level")
        
        # =====================================================
        # FEATURE 4: VIX Percentile (Historical Context)
        # =====================================================
        df['vix_percentile'] = df['vix'].rolling(window=252).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
        )
        print("✓ Feature 4: VIX percentile (252-day)")
        
        # =====================================================
        # FEATURE 5: Volatility of Volatility (VIX Std Dev)
        # =====================================================
        df['vix_volatility'] = df['vix'].rolling(window=20).std()
        print("✓ Feature 5: VIX volatility (20-day std)")
        
        # =====================================================
        # FEATURE 6: VIX 5-day Change (Short-term Regime)
        # =====================================================
        df['vix_change_5d'] = df['vix'].pct_change(5)
        print("✓ Feature 6: VIX 5-day change")
        
        # =====================================================
        # FEATURE 7: VIX 20-day Change (Medium-term Regime)
        # =====================================================
        df['vix_change_20d'] = df['vix'].pct_change(20)
        print("✓ Feature 7: VIX 20-day change")
        
        # =====================================================
        # FEATURE 8: Correlation Momentum (5-day)
        # =====================================================
        df['corr_momentum_5d'] = df['impl_corr'].diff(5)
        print("✓ Feature 8: Correlation momentum (5-day)")
        
        # =====================================================
        # FEATURE 9: Correlation Momentum (20-day)
        # =====================================================
        df['corr_momentum_20d'] = df['impl_corr'].diff(20)
        print("✓ Feature 9: Correlation momentum (20-day)")
        
        # =====================================================
        # FEATURE 10: Mean Reversion Distance (252-day MA)
        # =====================================================
        df['corr_ma_252'] = df['impl_corr'].rolling(window=252).mean()
        df['corr_mean_reversion'] = df['impl_corr'] - df['corr_ma_252']
        print("✓ Feature 10: Mean reversion distance (252-day)")
        
        # =====================================================
        # FEATURE 11: Correlation Percentile
        # =====================================================
        df['corr_percentile'] = df['impl_corr'].rolling(window=252).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
        )
        print("✓ Feature 11: Correlation percentile (252-day)")
        
        # =====================================================
        # FEATURE 12: Z-score Momentum (5-day)
        # =====================================================
        df['zscore_momentum_5d'] = df['z_score'].diff(5)
        print("✓ Feature 12: Z-score momentum (5-day)")
        
        # =====================================================
        # FEATURE 13: Index IV Level (QQQ/NDX IV)
        # =====================================================
        df['index_iv'] = df['ndx_iv']
        print("✓ Feature 13: Index IV level")
        
        # =====================================================
        # FEATURE 14: Index IV Change (5-day)
        # =====================================================
        df['index_iv_change_5d'] = df['ndx_iv'].pct_change(5)
        print("✓ Feature 14: Index IV change (5-day)")
        
        # =====================================================
        # FEATURE 15: VIX Term Structure (VIX vs VXN spread)
        # =====================================================
        if 'vxn_level' in df.columns:
            df['vix_vxn_spread'] = df['vix'] - df['vxn_level']
        else:
            df['vix_vxn_spread'] = 0  # Placeholder if VXN not available
        print("✓ Feature 15: VIX-VXN spread")
        
        # =====================================================
        # FEATURE 16: Rolling Sharpe of Z-score Signal
        # =====================================================
        # This captures recent signal quality
        df['zscore_rolling_mean'] = df['z_score'].rolling(window=20).mean()
        df['zscore_rolling_std'] = df['z_score'].rolling(window=20).std()
        print("✓ Feature 16: Z-score rolling statistics")
        
        # =====================================================
        # FEATURE 17: Days Since Last Extreme Z-score
        # =====================================================
        df['extreme_zscore'] = (df['z_score'].abs() > 1.5).astype(int)
        df['days_since_extreme'] = df['extreme_zscore'].groupby(
            (df['extreme_zscore'] != df['extreme_zscore'].shift()).cumsum()
        ).cumcount()
        print("✓ Feature 17: Days since last extreme Z-score")
        
        # =====================================================
        # FEATURE 18: VIX Regime (Low/Medium/High)
        # =====================================================
        df['vix_regime'] = pd.cut(
            df['vix'], 
            bins=[0, 15, 25, 100], 
            labels=[0, 1, 2]  # 0=Low, 1=Medium, 2=High
        ).astype(float)
        print("✓ Feature 18: VIX regime (categorical)")
        
        self.features_df = df
        print(f"\n✓ Total features created: 18")
        print(f"✓ Final dataframe: {len(df)} rows")
        print("=" * 60)
        
        return df
    
    def create_target_variable(self, holding_period: int = 5, 
                                pnl_multiplier: float = 7.0,
                                profit_threshold: float = 0.01) -> pd.DataFrame:
        """
        Create the THREE-CLASS target variable (y) for ML training.
        
        Classes:
            0 = SHORT_DISPERSION (correlation decreased significantly)
            1 = LONG_DISPERSION (correlation increased significantly)
            2 = NO_TRADE (no significant movement either direction)
        
        Args:
            holding_period: Number of days to hold the trade
            pnl_multiplier: Multiplier for P&L calculation (matches backtest)
            profit_threshold: Minimum correlation change to be considered profitable
            
        Returns:
            DataFrame with target variable added
        """
        print("\n" + "=" * 60)
        print("CREATING THREE-CLASS TARGET VARIABLE")
        print("=" * 60)
        
        df = self.features_df.copy()
        
        # Calculate future correlation (after holding period)
        df['future_corr'] = df['impl_corr'].shift(-holding_period)
        
        # Calculate correlation change
        df['corr_change'] = df['future_corr'] - df['impl_corr']
        
        # Calculate P&L for each direction
        df['short_pnl'] = -df['corr_change'] * pnl_multiplier  # SHORT profits when corr decreases
        df['long_pnl'] = df['corr_change'] * pnl_multiplier    # LONG profits when corr increases
        
        # THREE-CLASS TARGET:
        # 0 = SHORT_DISPERSION profitable (correlation decreased by > threshold)
        # 1 = LONG_DISPERSION profitable (correlation increased by > threshold)
        # 2 = NO_TRADE (neither direction clearly profitable)
        
        def classify_outcome(row):
            if pd.isna(row['corr_change']):
                return np.nan
            
            corr_change = row['corr_change']
            
            # SHORT is profitable when correlation DECREASES
            if corr_change < -profit_threshold:
                return 0  # SHORT_DISPERSION
            # LONG is profitable when correlation INCREASES
            elif corr_change > profit_threshold:
                return 1  # LONG_DISPERSION
            else:
                return 2  # NO_TRADE (sideways)
        
        df['target'] = df.apply(classify_outcome, axis=1)
        
        # Also keep binary targets for reference
        df['target_short'] = (df['corr_change'] < -profit_threshold).astype(int)
        df['target_long'] = (df['corr_change'] > profit_threshold).astype(int)
        df['target_pnl'] = -df['corr_change'] * pnl_multiplier
        
        # Calculate baseline statistics
        valid_targets = df['target'].dropna()
        print(f"✓ Three-class target variable created")
        print(f"  Total samples: {len(valid_targets)}")
        print(f"  Profit threshold: {profit_threshold:.2%} correlation change")
        print(f"\n  Class distribution:")
        
        class_counts = valid_targets.value_counts().sort_index()
        class_names = {0: 'SHORT_DISPERSION', 1: 'LONG_DISPERSION', 2: 'NO_TRADE'}
        for cls, count in class_counts.items():
            pct = count / len(valid_targets) * 100
            print(f"    {int(cls)} ({class_names[int(cls)]}): {count} ({pct:.1f}%)")
        
        # Calculate target for Z-score > 1.5 days only (current strategy)
        zscore_short_days = df[df['z_score'] > 1.5]['target'].dropna()
        if len(zscore_short_days) > 0:
            short_correct = (zscore_short_days == 0).sum()
            print(f"\n  For Z-score > 1.5 days (SHORT signals):")
            print(f"    Total signals: {len(zscore_short_days)}")
            print(f"    Correct (SHORT profitable): {short_correct} ({short_correct/len(zscore_short_days)*100:.1f}%)")
        
        zscore_long_days = df[df['z_score'] < -1.5]['target'].dropna()
        if len(zscore_long_days) > 0:
            long_correct = (zscore_long_days == 1).sum()
            print(f"\n  For Z-score < -1.5 days (LONG signals):")
            print(f"    Total signals: {len(zscore_long_days)}")
            print(f"    Correct (LONG profitable): {long_correct} ({long_correct/len(zscore_long_days)*100:.1f}%)")
        
        self.features_df = df
        print("=" * 60)
        
        return df
    
    def get_ml_dataset(self, min_zscore: float = None) -> tuple:
        """
        Get the final ML dataset ready for training.
        
        Args:
            min_zscore: If provided, only include days where |z_score| > min_zscore
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        df = self.features_df.copy()
        
        # Define feature columns
        feature_cols = [
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
        
        # Filter by z_score if specified
        if min_zscore is not None:
            df = df[df['z_score'].abs() > min_zscore]
        
        # Drop rows with NaN in features or target
        df_clean = df[feature_cols + ['target']].dropna()
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        
        return X, y, feature_cols
    
    def save_dataset(self, output_path: str = None):
        """Save the complete feature dataset to CSV."""
        if output_path is None:
            output_path = '/home/ubuntu/dispersion_live/ml_data/features_dataset.csv'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.features_df.to_csv(output_path)
        print(f"\n✓ Dataset saved to: {output_path}")
        print(f"  Rows: {len(self.features_df)}")
        print(f"  Columns: {len(self.features_df.columns)}")
        
        return output_path


def generate_ml_dataset():
    """
    Main function to generate the complete ML dataset.
    """
    print("=" * 70)
    print("ML FEATURE ENGINEERING FOR DISPERSION TRADING")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Load data
    engineer.load_data()
    
    # Create features
    engineer.create_features()
    
    # Create target variable
    engineer.create_target_variable(holding_period=5, pnl_multiplier=7.0)
    
    # Save dataset
    output_path = engineer.save_dataset()
    
    # Get ML-ready data
    X, y, feature_names = engineer.get_ml_dataset()
    
    print("\n" + "=" * 70)
    print("ML DATASET SUMMARY")
    print("=" * 70)
    print(f"Total samples: {len(X)}")
    print(f"Features: {len(feature_names)}")
    print(f"Target distribution: {y.mean()*100:.1f}% positive (profitable)")
    print("\nFeature names:")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
    print("=" * 70)
    
    return engineer, X, y, feature_names


if __name__ == "__main__":
    engineer, X, y, feature_names = generate_ml_dataset()
