# Dispersion Trading Live System

## Overview

This system generates daily trading signals for dispersion trading based on implied correlation mean-reversion. It connects to Interactive Brokers (IBKR) to fetch real-time implied volatility data and generates actionable trading signals.

**UPDATED December 2024:** Now uses **QQQ** instead of NDX for the index leg.

### Why QQQ Instead of NDX?

| Metric | NDX | QQQ | Advantage |
|--------|-----|-----|----------|
| Daily Return Correlation | - | 99.86% | Near-perfect tracking |
| IV Correlation | - | 98.94% | Near-perfect IV tracking |
| Signal Agreement | - | 91.05% | Same signals 91% of the time |
| Average Daily Volume | ~50,000 | ~2,000,000 | **40x better** |
| Bid-Ask Spread (ATM) | ~$2.00 | ~$0.05 | **97% tighter** |
| Contract Size | $100 × Index | $100 × ETF | **40x smaller** |

**Backtest Results (15 years, 2010-2025, QQQ-based):**
- Total Return: **+11.66%**
- Sharpe Ratio: **3.63**
- Max Drawdown: **-0.50%**
- Win Rate: **67.73%**

---

## System Components

### 1. `ibkr_connector.py`
Handles connection to IBKR TWS/Gateway and fetches real-time IV data.

### 2. `correlation_calculator.py`
Calculates implied correlation from IV data and generates Z-score signals.

### 3. `signal_generator.py`
Main signal generation script that orchestrates data fetching and signal calculation.

### 4. `paper_trader.py`
Paper trading execution module for tracking positions and P&L.

### 5. `run_daily.py`
Main entry point for daily trading operations.

---

## Quick Start

### Prerequisites

1. **IBKR Account** (Paper trading account is free)
2. **TWS or IB Gateway** installed and running
3. **Python 3.8+** with required packages

### Installation

```powershell
# Install required packages
pip install ibapi pandas numpy scipy matplotlib

# Extract the system
Expand-Archive -Path "Dispersion_Live_Trading_System.zip" -DestinationPath "." -Force
cd dispersion_live
```

### Setup IBKR

1. Open TWS or IB Gateway
2. Go to **File > Global Configuration > API > Settings**
3. Enable **"Enable ActiveX and Socket Clients"**
4. Set **Socket port** to `7497` (paper) or `7496` (live)
5. Uncheck **"Read-Only API"**

### Copy Historical Data

Copy the Bloomberg backtest correlation data for Z-score calculation:

```powershell
mkdir historical_data
copy ..\dispersion_bloomberg\results\correlation_data.csv historical_data\
```

---

## Usage

### Daily Signal Generation (with IBKR)

```powershell
python run_daily.py
```

This will:
1. Connect to IBKR
2. Fetch real-time IV for NDX and components
3. Calculate implied correlation
4. Generate Z-score signal
5. Execute paper trade if signal is actionable
6. Save results to files

### Offline Test Mode

```powershell
python run_daily.py --offline
```

Tests the system with sample data (no IBKR connection required).

### Signal Only (No Trading)

```powershell
python run_daily.py --signal-only
```

Generates signal but does not execute trades.

### Check Positions

```powershell
python run_daily.py --check-positions
```

Reviews open positions and closes any that have expired.

---

## Trading Logic

### Signal Generation

1. **Fetch IV Data**: Get implied volatility for QQQ (index proxy) and top 50 components
2. **Calculate Implied Correlation**: Use portfolio variance formula
3. **Calculate Z-Score**: Compare to 60-day rolling distribution
4. **Generate Signal**:
   - Z > 1.5 → **SHORT DISPERSION** (sell index vol, buy component vol)
   - Z < -1.5 → **LONG DISPERSION** (buy index vol, sell component vol)
   - Otherwise → **NO TRADE**

### Trade Structure

**SHORT DISPERSION** (when correlation is high):
- Sell QQQ straddle (50% allocation)
- Buy component straddles (50% allocation, split among top 10)

**LONG DISPERSION** (when correlation is low):
- Buy QQQ straddle (50% allocation)
- Sell component straddles (50% allocation, split among top 10)

### Position Management

- **Position Size**: 2% of capital per trade
- **Max Positions**: 3 concurrent
- **Holding Period**: 5 days
- **Exit**: Close after holding period expires

---

## Output Files

### Signals
- `signals/signal_YYYYMMDD.json` - Daily signal details
- `signals/signal_history.csv` - All historical signals

### Positions
- `positions/current_positions.json` - Open positions
- `positions/trade_history.csv` - Closed trade records

### Logs
- `logs/signal_YYYYMMDD.log` - Daily log files
- `logs/trader_YYYYMMDD.log` - Trading log files

### Historical Data
- `historical_data/correlation_data.csv` - Correlation history for Z-score

---

## Configuration

Edit parameters in each module:

### `ibkr_connector.py`
```python
TWS_HOST = "127.0.0.1"
TWS_PORT = 7497  # 7497=paper, 7496=live
CLIENT_ID = 1
```

### `correlation_calculator.py`
```python
LOOKBACK_DAYS = 60
Z_THRESHOLD = 1.5
MIN_COMPONENTS = 20
```

### `paper_trader.py`
```python
INITIAL_CAPITAL = 100000
POSITION_SIZE_PCT = 0.02  # 2%
MAX_POSITIONS = 3
HOLDING_PERIOD_DAYS = 5
NUM_COMPONENTS_TO_TRADE = 10
```

---

## Daily Workflow

### Morning (After Market Open)

1. **Start TWS/Gateway**
2. **Run signal generator**:
   ```powershell
   python run_daily.py
   ```
3. **Review signal** in `signals/signal_YYYYMMDD.json`
4. **Check alerts** for actionable signals

### End of Day

1. **Check positions**:
   ```powershell
   python run_daily.py --check-positions
   ```
2. **Review P&L** in portfolio summary

---

## Troubleshooting

### Connection Failed
- Ensure TWS/Gateway is running
- Check API settings are enabled
- Verify port number (7497 for paper)

### No IV Data
- Check market data subscriptions in IBKR
- Some stocks may not have options data

### Z-Score Calculation Failed
- Ensure `historical_data/correlation_data.csv` exists
- Need at least 60 days of historical data

---

## Risk Warnings

⚠️ **Paper Trading First**: Always test thoroughly with paper trading before live trading.

⚠️ **Position Sizing**: Start with small positions (1-2% of capital).

⚠️ **Market Conditions**: Strategy may underperform during extreme market conditions.

⚠️ **Execution Risk**: Real options trading involves bid-ask spreads and slippage.

---

## Support

For issues or questions, review:
1. Log files in `logs/` directory
2. Signal history in `signals/signal_history.csv`
3. Position records in `positions/`

---

*System Version: 2.0 (QQQ-based)*
*Last Updated: December 2025*
