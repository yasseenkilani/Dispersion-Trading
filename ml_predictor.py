"""
ML Predictor Module for Dispersion Trading - HYBRID VERSION
============================================================

This module implements a hybrid approach:
- SHORT signals: ML-enhanced (66.9% win rate vs 61.5% baseline)
- LONG signals: Z-score only (ML doesn't add value for LONG)

The ML model focuses on predicting SHORT_DISPERSION profitability,
while LONG_DISPERSION signals fall back to the proven Z-score threshold.

Usage:
    from ml_predictor import MLPredictor
    
    predictor = MLPredictor()
    prediction = predictor.predict(iv_data, z_score, impl_corr)
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MLPredictor:
    """
    Hybrid ML predictor: ML for SHORT signals, Z-score for LONG signals.
    
    This approach maximizes reliability by using ML only where it adds value.
    """
    
    # Thresholds
    ML_SHORT_THRESHOLD = 0.60  # ML probability threshold for SHORT
    ZSCORE_LONG_THRESHOLD = -1.5  # Z-score threshold for LONG (unchanged)
    
    def __init__(self, model_path: str = None):
        """
        Initialize the ML predictor.
        
        Args:
            model_path: Path to the trained model pickle file
        """
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), 'ml_data', 'best_model.pkl'
        )
        
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.model_name = None
        self.loaded = False
        
        # Try to load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk."""
        if not os.path.exists(self.model_path):
            print(f"⚠ ML model not found at {self.model_path}")
            print("  Run train_model.py first to train the model")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_cols = model_data['feature_cols']
            self.model_name = model_data['model_name']
            self.loaded = True
            
            return True
            
        except Exception as e:
            print(f"⚠ Error loading ML model: {e}")
            return False
    
    def _calculate_features(self, iv_data: dict, z_score: float, impl_corr: float,
                           historical_data: pd.DataFrame = None) -> dict:
        """
        Calculate ML features from current market data.
        """
        features = {}
        
        # Feature 1: Z-score (provided)
        features['z_score'] = float(z_score) if not pd.isna(z_score) else 0.0
        
        # Feature 2: Implied correlation level
        features['impl_corr_level'] = float(impl_corr) if not pd.isna(impl_corr) else 0.3
        
        # Feature 3: VIX level
        features['vix_level'] = float(iv_data.get('vix_level', iv_data.get('index_iv', 20)))
        
        # Features requiring historical data
        if historical_data is not None and len(historical_data) > 252:
            recent = historical_data.tail(252).copy()
            
            # VIX percentile
            vix_col = 'vix' if 'vix' in recent.columns else 'ndx_iv'
            if vix_col in recent.columns:
                current_vix = features['vix_level']
                vix_series = recent[vix_col].dropna()
                if len(vix_series) > 0:
                    features['vix_percentile'] = float((vix_series < current_vix).mean())
                else:
                    features['vix_percentile'] = 0.5
            else:
                features['vix_percentile'] = 0.5
            
            # VIX volatility
            if vix_col in recent.columns:
                vix_tail = recent[vix_col].tail(20).dropna()
                features['vix_volatility'] = float(vix_tail.std()) if len(vix_tail) > 1 else 2.0
            else:
                features['vix_volatility'] = 2.0
            
            # VIX changes
            if vix_col in recent.columns:
                vix_series = recent[vix_col].dropna()
                if len(vix_series) >= 5:
                    features['vix_change_5d'] = float(vix_series.iloc[-1] / vix_series.iloc[-5] - 1)
                else:
                    features['vix_change_5d'] = 0.0
                if len(vix_series) >= 20:
                    features['vix_change_20d'] = float(vix_series.iloc[-1] / vix_series.iloc[-20] - 1)
                else:
                    features['vix_change_20d'] = 0.0
            else:
                features['vix_change_5d'] = 0.0
                features['vix_change_20d'] = 0.0
            
            # Correlation momentum
            if 'impl_corr' in recent.columns:
                corr_series = recent['impl_corr'].dropna()
                if len(corr_series) >= 5:
                    features['corr_momentum_5d'] = float(impl_corr - corr_series.iloc[-5])
                else:
                    features['corr_momentum_5d'] = 0.0
                if len(corr_series) >= 20:
                    features['corr_momentum_20d'] = float(impl_corr - corr_series.iloc[-20])
                else:
                    features['corr_momentum_20d'] = 0.0
            else:
                features['corr_momentum_5d'] = 0.0
                features['corr_momentum_20d'] = 0.0
            
            # Mean reversion distance
            if 'impl_corr' in recent.columns:
                corr_series = recent['impl_corr'].dropna()
                if len(corr_series) > 0:
                    features['corr_mean_reversion'] = float(impl_corr - corr_series.mean())
                else:
                    features['corr_mean_reversion'] = 0.0
            else:
                features['corr_mean_reversion'] = 0.0
            
            # Correlation percentile
            if 'impl_corr' in recent.columns:
                corr_series = recent['impl_corr'].dropna()
                if len(corr_series) > 0:
                    features['corr_percentile'] = float((corr_series < impl_corr).mean())
                else:
                    features['corr_percentile'] = 0.5
            else:
                features['corr_percentile'] = 0.5
            
            # Z-score momentum
            if 'z_score' in recent.columns:
                z_series = recent['z_score'].dropna()
                if len(z_series) >= 5:
                    features['zscore_momentum_5d'] = float(z_score - z_series.iloc[-5])
                else:
                    features['zscore_momentum_5d'] = 0.0
            else:
                features['zscore_momentum_5d'] = 0.0
            
            # Z-score rolling stats
            if 'z_score' in recent.columns:
                z_tail = recent['z_score'].tail(20).dropna()
                if len(z_tail) > 1:
                    features['zscore_rolling_mean'] = float(z_tail.mean())
                    features['zscore_rolling_std'] = float(z_tail.std())
                else:
                    features['zscore_rolling_mean'] = float(z_score)
                    features['zscore_rolling_std'] = 1.0
            else:
                features['zscore_rolling_mean'] = float(z_score)
                features['zscore_rolling_std'] = 1.0
            
            # Days since extreme
            if 'z_score' in recent.columns:
                z_series = recent['z_score'].dropna()
                extreme_mask = z_series.abs() > 1.5
                if extreme_mask.any():
                    extreme_positions = np.where(extreme_mask.values)[0]
                    if len(extreme_positions) > 0:
                        features['days_since_extreme'] = len(z_series) - extreme_positions[-1] - 1
                    else:
                        features['days_since_extreme'] = len(z_series)
                else:
                    features['days_since_extreme'] = len(z_series)
            else:
                features['days_since_extreme'] = 30
        else:
            # Default values
            features['vix_percentile'] = 0.5
            features['vix_volatility'] = 2.0
            features['vix_change_5d'] = 0.0
            features['vix_change_20d'] = 0.0
            features['corr_momentum_5d'] = 0.0
            features['corr_momentum_20d'] = 0.0
            features['corr_mean_reversion'] = 0.0
            features['corr_percentile'] = 0.5
            features['zscore_momentum_5d'] = 0.0
            features['zscore_rolling_mean'] = float(z_score)
            features['zscore_rolling_std'] = 1.0
            features['days_since_extreme'] = 30
        
        # Index IV
        features['index_iv'] = float(iv_data.get('index_iv', 20))
        features['index_iv_change_5d'] = 0.0
        
        # VIX-VXN spread
        vix = features['vix_level']
        vxn = float(iv_data.get('vxn_level', vix))
        features['vix_vxn_spread'] = vix - vxn
        
        # VIX regime
        if vix < 15:
            features['vix_regime'] = 0
        elif vix < 25:
            features['vix_regime'] = 1
        else:
            features['vix_regime'] = 2
        
        return features
    
    def _get_short_probability(self, feature_vector: np.ndarray) -> float:
        """
        Get probability of SHORT_DISPERSION being profitable.
        
        Handles both 2-class and 3-class models.
        """
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        if len(probabilities) == 2:
            # Legacy 2-class model: [NOT_SHORT, SHORT]
            return float(probabilities[1])
        else:
            # 3-class model: [SHORT, LONG, NO_TRADE]
            return float(probabilities[0])
    
    def predict(self, iv_data: dict, z_score: float, impl_corr: float,
                historical_data: pd.DataFrame = None) -> dict:
        """
        Generate hybrid ML/Z-score prediction.
        
        Strategy:
        - SHORT: Use ML probability (threshold: 60%)
        - LONG: Use Z-score only (threshold: -1.5)
        
        Args:
            iv_data: Dictionary with index_iv, component_ivs, etc.
            z_score: Current Z-score from correlation calculator
            impl_corr: Current implied correlation
            historical_data: Historical correlation data (optional)
            
        Returns:
            Dictionary with prediction results
        """
        if not self.loaded:
            return {
                'ml_available': False,
                'ml_signal': 'ML_NOT_AVAILABLE',
                'ml_recommendation': 'Train model first (run train_model.py)',
                'short_probability': None,
                'long_signal': z_score < self.ZSCORE_LONG_THRESHOLD
            }
        
        try:
            # Calculate features
            features = self._calculate_features(iv_data, z_score, impl_corr, historical_data)
            
            # Build feature vector
            feature_values = []
            for col in self.feature_cols:
                val = features.get(col, 0.0)
                if pd.isna(val):
                    val = 0.0
                feature_values.append(float(val))
            
            feature_vector = np.array([feature_values])
            
            # Handle NaN
            if np.isnan(feature_vector).any():
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)
            
            # Scale features for logistic regression
            if self.scaler is not None and self.model_name == 'logistic_regression':
                feature_vector = self.scaler.transform(feature_vector)
            
            # Get SHORT probability from ML
            short_prob = self._get_short_probability(feature_vector)
            
            # Determine signals using hybrid approach
            ml_short_signal = short_prob >= self.ML_SHORT_THRESHOLD
            zscore_long_signal = z_score <= self.ZSCORE_LONG_THRESHOLD
            
            # Determine final ML recommendation
            if ml_short_signal:
                ml_signal = 'SHORT_DISPERSION'
                ml_confidence = short_prob
                ml_recommendation = f'ML recommends SHORT (prob: {short_prob:.1%})'
            elif zscore_long_signal:
                ml_signal = 'LONG_DISPERSION'
                ml_confidence = abs(z_score) / 3.0  # Normalize Z-score to ~probability
                ml_recommendation = f'Z-score recommends LONG (Z: {z_score:.2f})'
            else:
                ml_signal = 'NO_TRADE'
                ml_confidence = 1.0 - short_prob
                ml_recommendation = f'No trade signal (SHORT prob: {short_prob:.1%}, Z: {z_score:.2f})'
            
            return {
                'ml_available': True,
                'ml_signal': ml_signal,
                'ml_recommendation': ml_recommendation,
                'ml_model': self.model_name,
                'ml_confidence': ml_confidence,
                'short_probability': short_prob,
                'short_ml_trigger': ml_short_signal,
                'long_zscore_trigger': zscore_long_signal,
                'strategy_used': 'ML' if ml_signal == 'SHORT_DISPERSION' else 'Z-SCORE',
                'ml_features': features
            }
            
        except Exception as e:
            return {
                'ml_available': False,
                'ml_signal': 'ML_ERROR',
                'ml_recommendation': f'ML prediction error: {str(e)}',
                'short_probability': None
            }
    
    def get_signal_comparison(self, z_score: float, ml_result: dict) -> str:
        """
        Compare Z-score signal with ML prediction.
        
        Args:
            z_score: Current Z-score
            ml_result: Result from predict() method
            
        Returns:
            Comparison summary string
        """
        # Z-score signals (traditional)
        if z_score > 1.5:
            z_signal = 'SHORT_DISPERSION'
        elif z_score < -1.5:
            z_signal = 'LONG_DISPERSION'
        else:
            z_signal = 'NO_TRADE'
        
        # ML hybrid signal
        ml_signal = ml_result.get('ml_signal', 'NO_TRADE')
        short_prob = ml_result.get('short_probability', 0)
        strategy = ml_result.get('strategy_used', 'UNKNOWN')
        
        if z_signal == ml_signal:
            return f"✓ AGREEMENT: Both recommend {ml_signal}"
        else:
            if ml_signal == 'SHORT_DISPERSION':
                return f"⚠ ML EARLY SHORT: ML sees opportunity (prob: {short_prob:.1%}) Z-score hasn't triggered"
            elif ml_signal == 'LONG_DISPERSION':
                return f"✓ LONG via Z-score: Z={z_score:.2f} triggers LONG (ML not used for LONG)"
            else:
                return f"⚠ DIVERGENCE: Z-score={z_signal}, ML={ml_signal} (SHORT prob: {short_prob:.1%})"


def test_predictor():
    """Test the hybrid ML predictor with sample data."""
    print("=" * 60)
    print("ML PREDICTOR TEST (HYBRID: ML for SHORT, Z-score for LONG)")
    print("=" * 60)
    
    predictor = MLPredictor()
    
    if not predictor.loaded:
        print("Model not loaded. Run train_model.py first.")
        return
    
    print(f"Model loaded: {predictor.model_name}")
    print(f"Features: {len(predictor.feature_cols)}")
    print(f"SHORT threshold: {predictor.ML_SHORT_THRESHOLD:.0%}")
    print(f"LONG Z-score threshold: {predictor.ZSCORE_LONG_THRESHOLD}")
    
    # Test 1: High correlation (expect ML SHORT)
    print("\n" + "-" * 40)
    print("Test 1: High correlation (expect ML SHORT)")
    print("-" * 40)
    
    iv_data = {'index_iv': 22.5, 'vix_level': 18.0}
    z_score = 1.2  # Below Z-score threshold but ML might trigger
    impl_corr = 0.42
    
    result = predictor.predict(iv_data, z_score, impl_corr)
    print(f"  Z-score: {z_score}")
    print(f"  Impl Corr: {impl_corr}")
    print(f"  ML Signal: {result['ml_signal']}")
    print(f"  SHORT Probability: {result.get('short_probability', 0):.1%}")
    print(f"  Strategy Used: {result.get('strategy_used', 'N/A')}")
    print(f"  Recommendation: {result['ml_recommendation']}")
    
    # Test 2: Low correlation (expect Z-score LONG)
    print("\n" + "-" * 40)
    print("Test 2: Low correlation (expect Z-score LONG)")
    print("-" * 40)
    
    z_score = -1.8  # Below LONG threshold
    impl_corr = 0.18
    
    result = predictor.predict(iv_data, z_score, impl_corr)
    print(f"  Z-score: {z_score}")
    print(f"  Impl Corr: {impl_corr}")
    print(f"  ML Signal: {result['ml_signal']}")
    print(f"  SHORT Probability: {result.get('short_probability', 0):.1%}")
    print(f"  Strategy Used: {result.get('strategy_used', 'N/A')}")
    print(f"  Recommendation: {result['ml_recommendation']}")
    
    # Test 3: Neutral (expect NO_TRADE)
    print("\n" + "-" * 40)
    print("Test 3: Neutral (expect NO_TRADE)")
    print("-" * 40)
    
    z_score = 0.3
    impl_corr = 0.28
    
    result = predictor.predict(iv_data, z_score, impl_corr)
    print(f"  Z-score: {z_score}")
    print(f"  Impl Corr: {impl_corr}")
    print(f"  ML Signal: {result['ml_signal']}")
    print(f"  SHORT Probability: {result.get('short_probability', 0):.1%}")
    print(f"  Strategy Used: {result.get('strategy_used', 'N/A')}")
    print(f"  Recommendation: {result['ml_recommendation']}")
    
    # Test 4: ML early entry (Z-score not triggered but ML sees opportunity)
    print("\n" + "-" * 40)
    print("Test 4: ML early entry (Z < 1.5 but high SHORT prob)")
    print("-" * 40)
    
    z_score = 1.0  # Below 1.5 threshold
    impl_corr = 0.40  # Elevated correlation
    
    result = predictor.predict(iv_data, z_score, impl_corr)
    print(f"  Z-score: {z_score}")
    print(f"  Impl Corr: {impl_corr}")
    print(f"  ML Signal: {result['ml_signal']}")
    print(f"  SHORT Probability: {result.get('short_probability', 0):.1%}")
    print(f"  Strategy Used: {result.get('strategy_used', 'N/A')}")
    print(f"  Recommendation: {result['ml_recommendation']}")


if __name__ == "__main__":
    test_predictor()
