# Risk Management Guide - Dispersion Trading System v2

## Overview

This guide documents the risk management enhancements implemented in the dispersion trading system. The system now tracks **3 parallel portfolios** with different position sizing strategies to evaluate the effectiveness of Kelly criterion sizing and confidence-based allocation.

## Multi-Portfolio Tracking

### Portfolio Strategies

| Portfolio | Signal Source | Position Sizing | Purpose |
|-----------|---------------|-----------------|---------|
| **Fixed 2%** | Z-Score | Fixed 2% per trade | Baseline comparison |
| **Kelly 0.5x** | ML Hybrid | Half Kelly + confidence scaling | Conservative Kelly |
| **Kelly 1.0x** | ML Hybrid | Full Kelly + confidence scaling | Aggressive Kelly |

### Kelly Criterion Formula

The Kelly criterion determines optimal position size based on historical win rate:

```
Kelly % = (Win Rate × Win/Loss Ratio - Loss Rate) / Win/Loss Ratio

For 1:1 payoff ratio:
Kelly % = 2 × Win Rate - 1
```

With our historical win rate of **66.9%**:
- Full Kelly = 33.8%
- Half Kelly = 16.9%

### Confidence-Based Scaling

Position sizes are further adjusted based on ML model confidence:

| ML Probability | Multiplier | Rationale |
|----------------|------------|-----------|
| ≥85% | 1.50x | Very high confidence |
| ≥75% | 1.25x | High confidence |
| ≥65% | 1.00x | Medium confidence |
| ≥60% | 0.75x | Low confidence (at threshold) |
| <60% | 0.50x | Below threshold |

### Position Size Examples

For a $100,000 portfolio with 75% ML probability:

| Strategy | Kelly % | Confidence Mult | Final % | Amount |
|----------|---------|-----------------|---------|--------|
| Fixed 2% | 2.0% | 1.25x | 2.5% | $2,500 |
| Kelly 0.5x | 16.9% | 1.25x | 20.0%* | $20,000 |
| Kelly 1.0x | 33.8% | 1.25x | 20.0%* | $20,000 |

*Capped at 20% maximum position size

---

## Drawdown Controls

### Circuit Breakers

| Control | Threshold | Action |
|---------|-----------|--------|
| **Daily Loss** | -2% | Halt new trades for the day |
| **Weekly Loss** | -5% | Reduce position sizes by 50% |
| **Monthly Drawdown** | -10% | Halt all trading |
| **Consecutive Losses** | 5 trades | Reduce position sizes by 50% |

### How Controls Work

1. **Daily Loss Limit**: Calculated from daily P&L relative to base capital
2. **Weekly Loss Limit**: Sum of daily P&L for current week
3. **Monthly Drawdown**: Peak-to-trough drawdown from highest capital level
4. **Consecutive Losses**: Counter resets to 0 after any winning trade

### Risk Status Display

The system shows real-time risk status:

```
RISK STATUS
------------------------------------------------------------------------------------------
  [FIXED] 🟢 ACTIVE
    Drawdown: 0.0% | Daily P&L: $0 | Weekly P&L: $0
    Consecutive Losses: 0 | Size Reduction: 100%
  [KELLY_0.5X] 🟢 ACTIVE
    Drawdown: 0.0% | Daily P&L: $0 | Weekly P&L: $0
    Consecutive Losses: 0 | Size Reduction: 100%
  [KELLY_1.0X] 🔴 HALTED
    Drawdown: -10.5% | Daily P&L: -$500 | Weekly P&L: -$2,000
    Consecutive Losses: 3 | Size Reduction: 50%
    ⚠️  Monthly drawdown limit hit (-10.5%)
```

---

## Model Validation Results

### Purged K-Fold Cross-Validation

To prevent data leakage in time series, we use **purged K-fold** with a 20-day gap between train and test sets.

| Metric | Purged K-Fold | Walk-Forward |
|--------|---------------|--------------|
| Accuracy | 60.2% ± 1.8% | 61.6% ± 1.5% |
| AUC | 0.659 ± 0.013 | 0.669 ± 0.028 |
| Precision | 65.0% | 66.7% |
| Recall | 62.8% | 57.1% |

### Feature Importance Analysis

Top 10 most predictive features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | corr_mean_reversion | 12.1% |
| 2 | impl_corr_level | 11.5% |
| 3 | ndx_iv | 8.3% |
| 4 | index_iv | 8.2% |
| 5 | corr_momentum_20d | 5.4% |
| 6 | corr_percentile | 5.2% |
| 7 | vix_vxn_spread | 4.7% |
| 8 | vxn_level | 4.6% |
| 9 | vix_level | 3.9% |
| 10 | vix | 3.9% |

### Features to Consider Pruning (<3% importance)

- z_score (2.9%)
- vix_change_20d (2.9%)
- zscore_rolling_mean (2.7%)
- vix_volatility (2.3%)
- index_iv_change_5d (2.2%)
- corr_momentum_5d (2.1%)
- And 8 more...

---

## Usage

### Daily Run with Risk Management

```bash
# Full run with IBKR (recommended)
python run_daily_v2.py

# Offline test mode
python run_daily_v2.py --offline

# Signal only (no trades)
python run_daily_v2.py --signal-only

# Check positions and risk status
python run_daily_v2.py --check-positions

# Show risk configuration
python run_daily_v2.py --show-risk
```

### Run Model Validation

```bash
# Full validation pipeline
python model_validation.py
```

### Test Risk Manager

```bash
# Test Kelly calculations and drawdown controls
python risk_manager.py
```

---

## Files Added

| File | Description |
|------|-------------|
| `risk_manager.py` | Kelly criterion, confidence scaling, drawdown controls |
| `multi_portfolio_tracker.py` | Track 3 parallel portfolios |
| `model_validation.py` | Purged K-fold CV, feature importance |
| `run_daily_v2.py` | Integrated daily runner with risk management |
| `ml_data/feature_importance.png` | Feature importance visualization |
| `ml_data/validation_results.json` | CV results |

---

## Next Steps

1. **Monitor for 2-3 weeks**: Compare performance of Fixed vs Kelly strategies
2. **Feature pruning**: Remove low-importance features and retrain
3. **Hyperparameter tuning**: Use optimized parameters (C=1.0, balanced weights)
4. **Transition to vega-neutral**: After validation, implement vega-neutral sizing

---

*Last updated: January 2026*
