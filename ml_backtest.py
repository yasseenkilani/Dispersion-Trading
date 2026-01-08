"""
ML-Enhanced Dispersion Backtest
================================

This script performs a rigorous backtest comparing:
1. Baseline Z-score strategy (Z > 1.5)
2. ML-enhanced strategy (Probability > 0.65)

Uses walk-forward validation to avoid look-ahead bias.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Configuration
INITIAL_CAPITAL = 100000
POSITION_SIZE_PCT = 0.02
PNL_MULTIPLIER = 7
HOLDING_PERIOD = 5
Z_THRESHOLD = 1.5
ML_PROBABILITY_THRESHOLD = 0.65
TRAINING_WINDOW = 252 * 3  # 3 years of training data


def load_data():
    """Load the features dataset."""
    df = pd.read_csv('/home/ubuntu/dispersion_live/ml_data/features_dataset.csv', 
                     index_col=0, parse_dates=True)
    return df


def run_baseline_backtest(df):
    """Run the baseline Z-score backtest."""
    print("\n" + "=" * 60)
    print("BASELINE BACKTEST (Z-score > 1.5)")
    print("=" * 60)
    
    capital = INITIAL_CAPITAL
    position = 0
    entry_date = None
    entry_corr = None
    
    trades = []
    equity_curve = []
    
    dates = df.index.tolist()
    
    for i, date in enumerate(dates):
        row = df.loc[date]
        z_score = row['z_score']
        impl_corr = row['impl_corr']
        
        # Skip if NaN
        if pd.isna(z_score) or pd.isna(impl_corr):
            continue
        
        # Check for exit
        if position != 0 and entry_date is not None:
            days_held = (date - entry_date).days
            
            if days_held >= HOLDING_PERIOD:
                # Calculate P&L
                corr_change = impl_corr - entry_corr
                
                # SHORT_DISPERSION profits when correlation decreases
                pnl = -corr_change * capital * POSITION_SIZE_PCT * PNL_MULTIPLIER
                capital += pnl
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_corr': entry_corr,
                    'exit_corr': impl_corr,
                    'pnl': pnl,
                    'capital': capital
                })
                
                position = 0
                entry_date = None
        
        # Check for entry (SHORT_DISPERSION when Z > 1.5)
        if position == 0 and z_score > Z_THRESHOLD:
            position = -1
            entry_date = date
            entry_corr = impl_corr
        
        equity_curve.append({'date': date, 'capital': capital})
    
    return trades, equity_curve


def run_ml_backtest(df, feature_cols):
    """
    Run ML-enhanced backtest with walk-forward validation.
    
    Uses rolling training window to avoid look-ahead bias.
    """
    print("\n" + "=" * 60)
    print("ML-ENHANCED BACKTEST (Walk-Forward)")
    print("=" * 60)
    
    capital = INITIAL_CAPITAL
    position = 0
    entry_date = None
    entry_corr = None
    
    trades = []
    equity_curve = []
    
    dates = df.index.tolist()
    
    # Start after we have enough training data
    start_idx = TRAINING_WINDOW + 60  # Extra buffer for rolling features
    
    print(f"Training window: {TRAINING_WINDOW} days")
    print(f"Starting backtest from index {start_idx}")
    print(f"Backtest period: {dates[start_idx].date()} to {dates[-1].date()}")
    
    # Walk-forward loop
    model = None
    scaler = None
    last_train_date = None
    retrain_frequency = 63  # Retrain quarterly
    
    for i in range(start_idx, len(dates)):
        date = dates[i]
        row = df.loc[date]
        z_score = row['z_score']
        impl_corr = row['impl_corr']
        
        # Skip if NaN
        if pd.isna(z_score) or pd.isna(impl_corr):
            continue
        
        # Retrain model periodically
        if model is None or (last_train_date is not None and 
                            (date - last_train_date).days >= retrain_frequency):
            # Get training data (past TRAINING_WINDOW days)
            train_start = i - TRAINING_WINDOW
            train_end = i
            
            train_df = df.iloc[train_start:train_end].dropna(subset=feature_cols + ['target'])
            
            if len(train_df) > 100:  # Need enough training data
                X_train = train_df[feature_cols].values
                y_train = train_df['target'].values
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                last_train_date = date
        
        # Check for exit
        if position != 0 and entry_date is not None:
            days_held = (date - entry_date).days
            
            if days_held >= HOLDING_PERIOD:
                # Calculate P&L
                corr_change = impl_corr - entry_corr
                pnl = -corr_change * capital * POSITION_SIZE_PCT * PNL_MULTIPLIER
                capital += pnl
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_corr': entry_corr,
                    'exit_corr': impl_corr,
                    'pnl': pnl,
                    'capital': capital
                })
                
                position = 0
                entry_date = None
        
        # Check for entry using ML model
        if position == 0 and model is not None:
            # Get current features
            features = row[feature_cols].values.reshape(1, -1)
            
            if not np.any(np.isnan(features)):
                features_scaled = scaler.transform(features)
                prob = model.predict_proba(features_scaled)[0, 1]
                
                # Enter if probability > threshold
                if prob > ML_PROBABILITY_THRESHOLD:
                    position = -1
                    entry_date = date
                    entry_corr = impl_corr
        
        equity_curve.append({'date': date, 'capital': capital})
    
    return trades, equity_curve


def calculate_metrics(trades, equity_curve, name):
    """Calculate performance metrics."""
    if not trades:
        return {}
    
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index('date')
    
    # Total return
    total_return = (equity_df['capital'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    
    # Annualized return
    years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
    annual_return = ((equity_df['capital'].iloc[-1] / INITIAL_CAPITAL) ** (1/years) - 1) * 100
    
    # Daily returns for Sharpe
    equity_df['returns'] = equity_df['capital'].pct_change()
    daily_returns = equity_df['returns'].dropna()
    
    # Sharpe ratio
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Max drawdown
    rolling_max = equity_df['capital'].cummax()
    drawdown = (equity_df['capital'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Win rate
    wins = len(trades_df[trades_df['pnl'] > 0])
    win_rate = wins / len(trades_df) * 100
    
    # Profit factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'name': name,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(trades_df),
        'profit_factor': profit_factor,
        'final_capital': equity_df['capital'].iloc[-1]
    }


def main():
    """Run the full backtest comparison."""
    print("=" * 70)
    print("ML vs BASELINE DISPERSION BACKTEST")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} rows")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Feature columns
    feature_cols = [
        'z_score', 'impl_corr_level', 'vix_level', 'vix_percentile',
        'vix_volatility', 'vix_change_5d', 'vix_change_20d',
        'corr_momentum_5d', 'corr_momentum_20d', 'corr_mean_reversion',
        'corr_percentile', 'zscore_momentum_5d', 'index_iv',
        'index_iv_change_5d', 'vix_vxn_spread', 'zscore_rolling_mean',
        'zscore_rolling_std', 'days_since_extreme', 'vix_regime'
    ]
    
    # Run baseline backtest
    baseline_trades, baseline_equity = run_baseline_backtest(df)
    baseline_metrics = calculate_metrics(baseline_trades, baseline_equity, 'Baseline (Z>1.5)')
    
    # Run ML backtest
    ml_trades, ml_equity = run_ml_backtest(df, feature_cols)
    ml_metrics = calculate_metrics(ml_trades, ml_equity, 'ML-Enhanced')
    
    # Print comparison
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS COMPARISON")
    print("=" * 70)
    
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║              BASELINE vs ML-ENHANCED BACKTEST RESULTS                ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║                                                                      ║")
    print("║  ┌─────────────────┬──────────────┬──────────────┬────────────────┐  ║")
    print("║  │     Metric      │   BASELINE   │ ML-ENHANCED  │     Change     │  ║")
    print("║  ├─────────────────┼──────────────┼──────────────┼────────────────┤  ║")
    
    metrics_to_show = [
        ('Total Return', 'total_return', '%', '.1f'),
        ('Annual Return', 'annual_return', '%', '.1f'),
        ('Sharpe Ratio', 'sharpe', '', '.2f'),
        ('Max Drawdown', 'max_drawdown', '%', '.1f'),
        ('Win Rate', 'win_rate', '%', '.1f'),
        ('Total Trades', 'total_trades', '', '.0f'),
        ('Profit Factor', 'profit_factor', '', '.2f'),
        ('Final Capital', 'final_capital', '$', ',.0f'),
    ]
    
    for label, key, prefix, fmt in metrics_to_show:
        b_val = baseline_metrics.get(key, 0)
        m_val = ml_metrics.get(key, 0)
        
        if key in ['total_return', 'annual_return', 'sharpe', 'win_rate', 'profit_factor']:
            change = m_val - b_val
            change_str = f"+{change:{fmt}}" if change >= 0 else f"{change:{fmt}}"
        elif key == 'max_drawdown':
            change = m_val - b_val
            change_str = f"+{change:{fmt}}" if change >= 0 else f"{change:{fmt}}"
        elif key == 'final_capital':
            change = m_val - b_val
            change_str = f"${change:+,.0f}"
        else:
            change = m_val - b_val
            change_str = f"+{change:{fmt}}" if change >= 0 else f"{change:{fmt}}"
        
        if prefix == '$':
            b_str = f"${b_val:{fmt}}"
            m_str = f"${m_val:{fmt}}"
        elif prefix == '%':
            b_str = f"{b_val:{fmt}}%"
            m_str = f"{m_val:{fmt}}%"
        else:
            b_str = f"{b_val:{fmt}}"
            m_str = f"{m_val:{fmt}}"
        
        print(f"║  │ {label:<15} │ {b_str:>12} │ {m_str:>12} │ {change_str:>14} │  ║")
    
    print("║  └─────────────────┴──────────────┴──────────────┴────────────────┘  ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    improvements = 0
    if ml_metrics['sharpe'] > baseline_metrics['sharpe']:
        print("✓ Sharpe Ratio improved")
        improvements += 1
    else:
        print("✗ Sharpe Ratio decreased")
    
    if ml_metrics['total_return'] > baseline_metrics['total_return']:
        print("✓ Total Return improved")
        improvements += 1
    else:
        print("✗ Total Return decreased")
    
    if ml_metrics['max_drawdown'] > baseline_metrics['max_drawdown']:
        print("✓ Max Drawdown improved (less negative)")
        improvements += 1
    else:
        print("✗ Max Drawdown worse")
    
    if ml_metrics['win_rate'] > baseline_metrics['win_rate']:
        print("✓ Win Rate improved")
        improvements += 1
    else:
        print("✗ Win Rate decreased")
    
    print(f"\nImprovements: {improvements}/4")
    
    if improvements >= 3:
        print("\n✓ ML-ENHANCED STRATEGY IS RECOMMENDED")
    elif improvements >= 2:
        print("\n⚠ ML-ENHANCED STRATEGY SHOWS MIXED RESULTS")
    else:
        print("\n✗ BASELINE STRATEGY IS RECOMMENDED")
    
    print("=" * 70)
    
    return baseline_metrics, ml_metrics


if __name__ == "__main__":
    baseline_metrics, ml_metrics = main()
