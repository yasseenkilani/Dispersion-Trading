"""
Live Implied Correlation Calculator
====================================

Calculates implied correlation from real-time IV data and compares
to historical distribution to generate Z-score signals.

Uses the same methodology as the backtest:
- Calculate implied correlation from index IV and component IVs
- Compare to rolling historical distribution
- Generate Z-score for trading signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Historical data path (from Bloomberg backtest)
HISTORICAL_DATA_PATH = "historical_data"

# Z-score parameters
LOOKBACK_DAYS = 60
Z_THRESHOLD = 1.5

# Minimum components required for valid calculation
MIN_COMPONENTS = 20

# =============================================================================
# IMPLIED CORRELATION CALCULATOR
# =============================================================================

class ImpliedCorrelationCalculator:
    """
    Calculates implied correlation and generates trading signals.
    """
    
    def __init__(self, historical_data_path=HISTORICAL_DATA_PATH):
        self.historical_data_path = historical_data_path
        self.historical_correlations = None
        self.load_historical_data()
    
    def load_historical_data(self):
        """Load historical correlation data from backtest."""
        corr_file = os.path.join(self.historical_data_path, "correlation_data.csv")
        
        if os.path.exists(corr_file):
            self.historical_correlations = pd.read_csv(corr_file)
            self.historical_correlations['date'] = pd.to_datetime(self.historical_correlations['date'])
            print(f"✓ Loaded {len(self.historical_correlations)} days of historical correlation data")
            print(f"  Date range: {self.historical_correlations['date'].min().date()} to {self.historical_correlations['date'].max().date()}")
            print(f"  Mean correlation: {self.historical_correlations['impl_corr'].mean():.4f}")
        else:
            print(f"⚠ Historical data not found at {corr_file}")
            print("  Will use live data only for Z-score calculation")
            self.historical_correlations = pd.DataFrame()
    
    def calculate_implied_correlation(self, ndx_iv, component_ivs, weights):
        """
        Calculate implied correlation from current IV data.
        
        Formula:
        σ²_index = Σ(w_i² * σ_i²) + 2 * ρ * Σ(w_i * w_j * σ_i * σ_j)
        
        Solving for ρ:
        ρ = (σ²_index - Σ(w_i² * σ_i²)) / (2 * Σ(w_i * w_j * σ_i * σ_j))
        
        Args:
            ndx_iv: NDX implied volatility (%)
            component_ivs: Dict of {ticker: iv_value}
            weights: Dict of {ticker: weight}
            
        Returns:
            Implied correlation (0 to 1) or None if insufficient data
        """
        # Get common tickers
        common_tickers = set(component_ivs.keys()) & set(weights.keys())
        
        if len(common_tickers) < MIN_COMPONENTS:
            print(f"  Warning: Only {len(common_tickers)} components available (need {MIN_COMPONENTS})")
            return None
        
        # Normalize weights to sum to 1
        total_weight = sum(weights[t] for t in common_tickers)
        norm_weights = {t: weights[t] / total_weight for t in common_tickers}
        
        # Convert IV from percentage to decimal
        sigma_index = ndx_iv / 100
        sigmas = {t: component_ivs[t] / 100 for t in common_tickers}
        
        # Calculate variance terms
        # Σ(w_i² * σ_i²)
        weighted_var_sum = sum(norm_weights[t]**2 * sigmas[t]**2 for t in common_tickers)
        
        # Σ(w_i * w_j * σ_i * σ_j) for i != j
        cross_term = 0
        tickers_list = list(common_tickers)
        for i, t1 in enumerate(tickers_list):
            for t2 in tickers_list[i+1:]:
                cross_term += norm_weights[t1] * norm_weights[t2] * sigmas[t1] * sigmas[t2]
        
        # Solve for implied correlation
        if cross_term <= 0:
            return None
        
        implied_corr = (sigma_index**2 - weighted_var_sum) / (2 * cross_term)
        
        # Clip to valid range [0, 1]
        implied_corr = np.clip(implied_corr, 0, 1)
        
        return implied_corr
    
    def calculate_z_score(self, current_corr, lookback_days=LOOKBACK_DAYS):
        """
        Calculate Z-score of current correlation vs historical distribution.
        
        Args:
            current_corr: Current implied correlation
            lookback_days: Number of days for rolling statistics
            
        Returns:
            Z-score value
        """
        if self.historical_correlations is None or len(self.historical_correlations) < lookback_days:
            print("  Warning: Insufficient historical data for Z-score")
            return None
        
        # Get recent historical correlations
        recent_data = self.historical_correlations.tail(lookback_days)
        
        mean_corr = recent_data['impl_corr'].mean()
        std_corr = recent_data['impl_corr'].std()
        
        if std_corr <= 0:
            return None
        
        z_score = (current_corr - mean_corr) / std_corr
        
        return z_score
    
    def generate_signal(self, ndx_iv, component_ivs, weights):
        """
        Generate trading signal from current IV data.
        
        Args:
            ndx_iv: NDX implied volatility (%)
            component_ivs: Dict of {ticker: iv_value}
            weights: Dict of {ticker: weight}
            
        Returns:
            Dict with signal details
        """
        print("\n" + "=" * 50)
        print("GENERATING DISPERSION SIGNAL")
        print("=" * 50)
        
        # Calculate implied correlation
        impl_corr = self.calculate_implied_correlation(ndx_iv, component_ivs, weights)
        
        if impl_corr is None:
            return {
                'timestamp': datetime.now(),
                'signal': 'NO_SIGNAL',
                'reason': 'Insufficient data for correlation calculation',
                'impl_corr': None,
                'z_score': None
            }
        
        print(f"\n  Implied Correlation: {impl_corr:.4f}")
        
        # Calculate Z-score
        z_score = self.calculate_z_score(impl_corr)
        
        if z_score is None:
            return {
                'timestamp': datetime.now(),
                'signal': 'NO_SIGNAL',
                'reason': 'Insufficient data for Z-score calculation',
                'impl_corr': impl_corr,
                'z_score': None
            }
        
        print(f"  Z-Score: {z_score:.4f}")
        
        # Generate signal
        if z_score > Z_THRESHOLD:
            signal = 'SHORT_DISPERSION'
            reason = f'High correlation (Z={z_score:.2f} > {Z_THRESHOLD})'
        elif z_score < -Z_THRESHOLD:
            signal = 'LONG_DISPERSION'
            reason = f'Low correlation (Z={z_score:.2f} < -{Z_THRESHOLD})'
        else:
            signal = 'NO_TRADE'
            reason = f'Z-score within threshold (|{z_score:.2f}| < {Z_THRESHOLD})'
        
        print(f"\n  Signal: {signal}")
        print(f"  Reason: {reason}")
        
        # Get historical context
        if len(self.historical_correlations) > 0:
            hist_mean = self.historical_correlations['impl_corr'].mean()
            hist_std = self.historical_correlations['impl_corr'].std()
            percentile = (self.historical_correlations['impl_corr'] < impl_corr).mean() * 100
            
            print(f"\n  Historical Context:")
            print(f"    Mean: {hist_mean:.4f}")
            print(f"    Std: {hist_std:.4f}")
            print(f"    Current Percentile: {percentile:.1f}%")
        
        return {
            'timestamp': datetime.now(),
            'signal': signal,
            'reason': reason,
            'impl_corr': impl_corr,
            'z_score': z_score,
            'ndx_iv': ndx_iv,
            'num_components': len(component_ivs)
        }
    
    def update_historical_data(self, new_corr, date=None):
        """
        Add new correlation data point to historical data.
        
        Args:
            new_corr: New implied correlation value
            date: Date for the data point (default: today)
        """
        if date is None:
            date = datetime.now().date()
        
        new_row = pd.DataFrame([{
            'date': pd.Timestamp(date),
            'impl_corr': new_corr
        }])
        
        if self.historical_correlations is None or len(self.historical_correlations) == 0:
            self.historical_correlations = new_row
        else:
            self.historical_correlations = pd.concat([self.historical_correlations, new_row], ignore_index=True)
        
        # Save updated data
        os.makedirs(self.historical_data_path, exist_ok=True)
        self.historical_correlations.to_csv(
            os.path.join(self.historical_data_path, "correlation_data.csv"),
            index=False
        )
        print(f"✓ Updated historical data with correlation {new_corr:.4f}")


# =============================================================================
# STANDALONE TEST
# =============================================================================

def test_calculator():
    """Test the correlation calculator with sample data."""
    print("=" * 60)
    print("IMPLIED CORRELATION CALCULATOR TEST")
    print("=" * 60)
    
    # Create calculator
    calc = ImpliedCorrelationCalculator()
    
    # Sample data (simulating real-time IV)
    ndx_iv = 18.5  # NDX IV at 18.5%
    
    component_ivs = {
        'NVDA': 45.2, 'AAPL': 22.1, 'GOOGL': 25.3, 'GOOG': 25.1,
        'MSFT': 21.5, 'AMZN': 28.7, 'META': 32.4, 'AVGO': 35.6,
        'TSLA': 55.2, 'NFLX': 38.9, 'ASML': 30.2, 'COST': 18.5,
        'AMD': 48.3, 'CSCO': 19.2, 'MU': 42.1, 'TMUS': 22.8,
        'AMAT': 35.4, 'PEP': 15.2, 'LRCX': 38.7, 'ISRG': 25.6,
        'QCOM': 32.1, 'INTU': 28.4, 'INTC': 35.8, 'BKNG': 30.2,
        'AMGN': 20.5, 'TXN': 25.3, 'KLAC': 36.2, 'PDD': 52.1,
        'GILD': 22.4, 'ADBE': 30.8,
    }
    
    weights = {
        'NVDA': 0.1213, 'AAPL': 0.1147, 'GOOGL': 0.1055, 'GOOG': 0.1055,
        'MSFT': 0.0999, 'AMZN': 0.0674, 'META': 0.0462, 'AVGO': 0.0454,
        'TSLA': 0.0448, 'NFLX': 0.0121, 'ASML': 0.0120, 'COST': 0.0108,
        'AMD': 0.0096, 'CSCO': 0.0088, 'MU': 0.0076, 'TMUS': 0.0062,
        'AMAT': 0.0059, 'PEP': 0.0059, 'LRCX': 0.0058, 'ISRG': 0.0056,
        'QCOM': 0.0054, 'INTU': 0.0052, 'INTC': 0.0051, 'BKNG': 0.0050,
        'AMGN': 0.0050, 'TXN': 0.0046, 'KLAC': 0.0046, 'PDD': 0.0044,
        'GILD': 0.0042, 'ADBE': 0.0042,
    }
    
    # Generate signal
    signal = calc.generate_signal(ndx_iv, component_ivs, weights)
    
    print("\n" + "=" * 50)
    print("SIGNAL RESULT")
    print("=" * 50)
    for key, value in signal.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_calculator()
