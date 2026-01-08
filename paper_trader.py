"""
Dispersion Trading Paper Trader
================================

Executes dispersion trades on IBKR paper trading account.

UPDATED: Now tracks both Z-score and ML strategies in parallel.

Trade Structure:
- SHORT DISPERSION: Sell QQQ straddle + Buy component straddles
- LONG DISPERSION: Buy QQQ straddle + Sell component straddles

This module handles:
1. Position sizing
2. Option contract selection (ATM, nearest expiry)
3. Order execution
4. Position tracking (separate portfolios for Z-score and ML)
5. P&L monitoring
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ibkr_connector import IBKRConnector, NDX_COMPONENTS, INDEX_SYMBOL

# =============================================================================
# CONFIGURATION
# =============================================================================

# Trading parameters
INITIAL_CAPITAL = 100000
POSITION_SIZE_PCT = 0.02  # 2% of capital per trade
MAX_POSITIONS = 3  # Maximum concurrent positions
HOLDING_PERIOD_DAYS = 5

# Option parameters
DAYS_TO_EXPIRY_TARGET = 30  # Target ~30 days to expiry
DELTA_TARGET = 0.50  # ATM options

# Top components to trade (by weight)
NUM_COMPONENTS_TO_TRADE = 30

# Paths
POSITIONS_DIR = "positions"
LOG_DIR = "logs"

# =============================================================================
# PAPER TRADER CLASS
# =============================================================================

class DispersionPaperTrader:
    """
    Executes dispersion trades on IBKR paper account.
    
    Supports dual portfolio tracking for Z-score and ML strategies.
    """
    
    def __init__(self, strategy_name="zscore", capital=INITIAL_CAPITAL):
        """
        Initialize paper trader.
        
        Args:
            strategy_name: 'zscore' or 'ml' to track different strategies
            capital: Initial capital for this strategy
        """
        self.strategy_name = strategy_name
        self.capital = capital
        self.positions = []
        self.trade_history = []
        self.ibkr = None
        self.logger = self._setup_logging()
        
        # Create directories
        os.makedirs(POSITIONS_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Strategy-specific file paths
        self.positions_file = os.path.join(POSITIONS_DIR, f"{strategy_name}_positions.json")
        self.trades_file = os.path.join(POSITIONS_DIR, f"{strategy_name}_trades.csv")
        self.capital_file = os.path.join(POSITIONS_DIR, f"{strategy_name}_capital.json")
        
        # Load existing positions and capital
        self._load_positions()
        self._load_capital()
    
    def _setup_logging(self):
        """Set up logging."""
        log_file = os.path.join(LOG_DIR, f"trader_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _load_positions(self):
        """Load existing positions from file."""
        if os.path.exists(self.positions_file):
            with open(self.positions_file, 'r') as f:
                self.positions = json.load(f)
            self.logger.info(f"[{self.strategy_name.upper()}] Loaded {len(self.positions)} existing positions")
    
    def _save_positions(self):
        """Save positions to file."""
        with open(self.positions_file, 'w') as f:
            json.dump(self.positions, f, indent=2, default=str)
    
    def _load_capital(self):
        """Load capital from file."""
        if os.path.exists(self.capital_file):
            with open(self.capital_file, 'r') as f:
                data = json.load(f)
                self.capital = data.get('capital', INITIAL_CAPITAL)
    
    def _save_capital(self):
        """Save capital to file."""
        with open(self.capital_file, 'w') as f:
            json.dump({'capital': self.capital, 'updated': datetime.now().isoformat()}, f)
    
    def _save_trade(self, trade):
        """Save trade to history."""
        trade_df = pd.DataFrame([trade])
        
        if os.path.exists(self.trades_file):
            history = pd.read_csv(self.trades_file)
            history = pd.concat([history, trade_df], ignore_index=True)
        else:
            history = trade_df
        
        history.to_csv(self.trades_file, index=False)
    
    def connect(self):
        """Connect to IBKR."""
        self.ibkr = IBKRConnector()
        return self.ibkr.connect()
    
    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ibkr:
            self.ibkr.disconnect()
    
    def get_position_size(self):
        """Calculate position size based on capital."""
        return self.capital * POSITION_SIZE_PCT
    
    def can_open_position(self):
        """Check if we can open a new position."""
        active_positions = [p for p in self.positions if p['status'] == 'OPEN']
        
        if len(active_positions) >= MAX_POSITIONS:
            return False
        
        return True
    
    def get_top_components(self, n=NUM_COMPONENTS_TO_TRADE):
        """Get top N components by weight."""
        sorted_components = sorted(NDX_COMPONENTS, key=lambda x: x[1], reverse=True)
        return sorted_components[:n]
    
    def calculate_trade_structure(self, signal_type, position_size):
        """Calculate the trade structure for a dispersion trade."""
        top_components = self.get_top_components()
        
        # Allocate 50% to index (QQQ), 50% to components
        index_allocation = position_size * 0.5
        component_allocation = position_size * 0.5
        
        # Per-component allocation
        per_component = component_allocation / len(top_components)
        
        if signal_type == 'SHORT_DISPERSION':
            index_side = 'SELL'
            component_side = 'BUY'
        else:
            index_side = 'BUY'
            component_side = 'SELL'
        
        trade_structure = {
            'signal_type': signal_type,
            'position_size': position_size,
            'index': {
                'symbol': INDEX_SYMBOL,
                'side': index_side,
                'allocation': index_allocation,
                'trade_type': 'STRADDLE',
                'option_type': 'ETF'
            },
            'components': []
        }
        
        for symbol, weight in top_components:
            trade_structure['components'].append({
                'symbol': symbol,
                'weight': weight,
                'side': component_side,
                'allocation': per_component,
                'trade_type': 'STRADDLE',
                'option_type': 'STOCK'
            })
        
        return trade_structure
    
    def execute_paper_trade(self, signal):
        """Execute a paper trade based on signal."""
        if signal['signal'] not in ['LONG_DISPERSION', 'SHORT_DISPERSION']:
            return None
        
        if not self.can_open_position():
            self.logger.warning(f"[{self.strategy_name.upper()}] Maximum positions reached")
            return None
        
        # Calculate position size
        position_size = self.get_position_size()
        
        # Calculate trade structure
        trade_structure = self.calculate_trade_structure(signal['signal'], position_size)
        
        # Create position record
        position = {
            'id': len(self.positions) + 1,
            'strategy': self.strategy_name,
            'entry_date': datetime.now().isoformat(),
            'exit_date': None,
            'signal_type': signal['signal'],
            'index_symbol': INDEX_SYMBOL,
            'entry_corr': signal.get('impl_corr'),
            'entry_z_score': signal.get('z_score'),
            'entry_ml_prob': signal.get('ml_probability'),
            'exit_corr': None,
            'exit_z_score': None,
            'position_size': position_size,
            'trade_structure': trade_structure,
            'status': 'OPEN',
            'pnl': None,
            'holding_period': HOLDING_PERIOD_DAYS
        }
        
        # Add to positions
        self.positions.append(position)
        self._save_positions()
        
        return position
    
    def _count_trading_days(self, start_date, end_date):
        """
        Count trading days between two dates (excludes weekends).
        
        Note: Does not account for market holidays - a minor simplification.
        """
        trading_days = 0
        current = start_date
        
        while current < end_date:
            # Monday = 0, Sunday = 6
            if current.weekday() < 5:  # Monday to Friday
                trading_days += 1
            current += timedelta(days=1)
        
        return trading_days
    
    def check_exit_conditions(self, current_corr=None, current_z=None):
        """Check if any positions should be closed based on trading days held."""
        positions_to_close = []
        
        for position in self.positions:
            if position['status'] != 'OPEN':
                continue
            
            entry_date = datetime.fromisoformat(position['entry_date'])
            
            # Count trading days (Mon-Fri only)
            trading_days_held = self._count_trading_days(entry_date, datetime.now())
            
            if trading_days_held >= position['holding_period']:
                positions_to_close.append(position)
        
        return positions_to_close
    
    def close_position(self, position, exit_corr=None, exit_z=None):
        """Close a position and calculate P&L."""
        position['exit_date'] = datetime.now().isoformat()
        position['exit_corr'] = exit_corr
        position['exit_z_score'] = exit_z
        position['status'] = 'CLOSED'
        
        # Calculate P&L based on correlation change
        if position['entry_corr'] is not None and exit_corr is not None:
            corr_change = exit_corr - position['entry_corr']
            
            VEGA_MULTIPLIER = 0.15
            
            if position['signal_type'] == 'SHORT_DISPERSION':
                pnl = -corr_change * position['position_size'] * VEGA_MULTIPLIER
            else:
                pnl = corr_change * position['position_size'] * VEGA_MULTIPLIER
            
            position['pnl'] = pnl
        else:
            position['pnl'] = 0
        
        # Update capital
        self.capital += position['pnl']
        
        # Save updated positions and capital
        self._save_positions()
        self._save_capital()
        
        # Save trade to history
        self._save_trade({
            'id': position['id'],
            'strategy': self.strategy_name,
            'entry_date': position['entry_date'],
            'exit_date': position['exit_date'],
            'signal_type': position['signal_type'],
            'index_symbol': position.get('index_symbol', INDEX_SYMBOL),
            'entry_corr': position['entry_corr'],
            'exit_corr': position['exit_corr'],
            'entry_ml_prob': position.get('entry_ml_prob'),
            'position_size': position['position_size'],
            'pnl': position['pnl']
        })
        
        return position
    
    def get_portfolio_stats(self):
        """Get portfolio statistics."""
        open_positions = [p for p in self.positions if p['status'] == 'OPEN']
        closed_positions = [p for p in self.positions if p['status'] == 'CLOSED']
        
        total_pnl = sum(p.get('pnl', 0) or 0 for p in closed_positions)
        wins = sum(1 for p in closed_positions if (p.get('pnl') or 0) > 0)
        win_rate = wins / len(closed_positions) * 100 if closed_positions else 0
        
        return {
            'strategy': self.strategy_name,
            'capital': self.capital,
            'open_positions': len(open_positions),
            'closed_positions': len(closed_positions),
            'total_pnl': total_pnl,
            'win_rate': win_rate
        }
    
    def print_portfolio_summary(self):
        """Print current portfolio summary."""
        stats = self.get_portfolio_stats()
        
        print(f"\n  [{self.strategy_name.upper()} STRATEGY]")
        print(f"  Capital: ${stats['capital']:,.2f}")
        print(f"  Open: {stats['open_positions']} | Closed: {stats['closed_positions']}")
        
        if stats['closed_positions'] > 0:
            print(f"  Total P&L: ${stats['total_pnl']:,.2f} | Win Rate: {stats['win_rate']:.1f}%")
        
        # Show open positions with trading days held
        open_positions = [p for p in self.positions if p['status'] == 'OPEN']
        if open_positions:
            print(f"  Open Positions:")
            for pos in open_positions:
                entry_date = datetime.fromisoformat(pos['entry_date'])
                trading_days = self._count_trading_days(entry_date, datetime.now())
                remaining = pos['holding_period'] - trading_days
                print(f"    #{pos['id']}: {pos['signal_type']} | Entry: {pos['entry_corr']:.4f} | Trading Days: {trading_days}/5 | Closes in: {remaining} days")


# =============================================================================
# DUAL PORTFOLIO MANAGER
# =============================================================================

class DualPortfolioManager:
    """
    Manages both Z-score and ML strategy portfolios in parallel.
    """
    
    def __init__(self):
        self.zscore_trader = DispersionPaperTrader(strategy_name="zscore")
        self.ml_trader = DispersionPaperTrader(strategy_name="ml")
    
    def execute_trades(self, signal, ml_prediction):
        """
        Execute trades for both strategies based on their respective signals.
        
        Args:
            signal: Z-score based signal dict
            ml_prediction: ML prediction dict with ml_signal and ml_probability
        """
        results = {
            'zscore': {'traded': False, 'position': None},
            'ml': {'traded': False, 'position': None}
        }
        
        # Check and close expired positions for both strategies
        self._check_and_close_positions(signal)
        
        # Execute Z-score trade
        if signal['signal'] in ['LONG_DISPERSION', 'SHORT_DISPERSION']:
            position = self.zscore_trader.execute_paper_trade(signal)
            if position:
                results['zscore']['traded'] = True
                results['zscore']['position'] = position
        
        # Execute ML trade (hybrid: ML for SHORT, Z-score for LONG)
        ml_signal = ml_prediction.get('ml_signal', 'NO_TRADE')
        if ml_signal in ['SHORT_DISPERSION', 'LONG_DISPERSION']:
            # Create ML signal dict
            ml_trade_signal = {
                'signal': ml_signal,
                'impl_corr': signal.get('impl_corr'),
                'z_score': signal.get('z_score'),
                'short_probability': ml_prediction.get('short_probability'),
                'strategy_used': ml_prediction.get('strategy_used', 'HYBRID')
            }
            position = self.ml_trader.execute_paper_trade(ml_trade_signal)
            if position:
                results['ml']['traded'] = True
                results['ml']['position'] = position
        
        return results
    
    def _check_and_close_positions(self, signal):
        """Check and close expired positions for both strategies."""
        exit_corr = signal.get('impl_corr')
        exit_z = signal.get('z_score')
        
        # Z-score positions
        for position in self.zscore_trader.check_exit_conditions():
            self.zscore_trader.close_position(position, exit_corr, exit_z)
        
        # ML positions
        for position in self.ml_trader.check_exit_conditions():
            self.ml_trader.close_position(position, exit_corr, exit_z)
    
    def print_comparison(self):
        """Print side-by-side comparison of both strategies."""
        z_stats = self.zscore_trader.get_portfolio_stats()
        ml_stats = self.ml_trader.get_portfolio_stats()
        
        print("\n" + "=" * 70)
        print("PORTFOLIO COMPARISON")
        print("=" * 70)
        
        print(f"\n┌{'─' * 68}┐")
        print(f"│{'STRATEGY PERFORMANCE':^68}│")
        print(f"├{'─' * 34}┬{'─' * 33}┤")
        print(f"│{'Z-SCORE STRATEGY':^34}│{'ML STRATEGY':^33}│")
        print(f"├{'─' * 34}┼{'─' * 33}┤")
        print(f"│ Capital: ${z_stats['capital']:>20,.2f} │ Capital: ${ml_stats['capital']:>19,.2f} │")
        print(f"│ Open Positions: {z_stats['open_positions']:>14} │ Open Positions: {ml_stats['open_positions']:>13} │")
        print(f"│ Closed Positions: {z_stats['closed_positions']:>12} │ Closed Positions: {ml_stats['closed_positions']:>11} │")
        print(f"│ Total P&L: ${z_stats['total_pnl']:>19,.2f} │ Total P&L: ${ml_stats['total_pnl']:>18,.2f} │")
        print(f"│ Win Rate: {z_stats['win_rate']:>20.1f}% │ Win Rate: {ml_stats['win_rate']:>19.1f}% │")
        print(f"└{'─' * 34}┴{'─' * 33}┘")
        
        # Performance difference
        if z_stats['capital'] != INITIAL_CAPITAL or ml_stats['capital'] != INITIAL_CAPITAL:
            z_return = (z_stats['capital'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            ml_return = (ml_stats['capital'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            
            print(f"\n  Returns: Z-Score {z_return:+.2f}% | ML {ml_return:+.2f}%")
            
            if ml_return > z_return:
                print(f"  → ML outperforming by {ml_return - z_return:.2f}%")
            elif z_return > ml_return:
                print(f"  → Z-Score outperforming by {z_return - ml_return:.2f}%")
            else:
                print(f"  → Strategies performing equally")
        
        print("=" * 70)


# =============================================================================
# TRADING SESSION
# =============================================================================

def run_trading_session(signal, ml_prediction=None):
    """
    Run a complete trading session with both strategies.
    
    Args:
        signal: Z-score based signal
        ml_prediction: ML prediction dict (optional)
    """
    manager = DualPortfolioManager()
    
    print("\n" + "=" * 70)
    print(f"TRADING SESSION - {INDEX_SYMBOL}")
    print("=" * 70)
    
    # Default ML prediction if not provided
    if ml_prediction is None:
        ml_prediction = {'ml_signal': 'NO_TRADE', 'ml_probability': None}
    
    # Execute trades for both strategies
    results = manager.execute_trades(signal, ml_prediction)
    
    # Print trade execution results
    print("\n" + "-" * 70)
    print("TRADE EXECUTION")
    print("-" * 70)
    
    z_signal = signal.get('signal', 'NO_TRADE')
    ml_signal = ml_prediction.get('ml_signal', 'NO_TRADE')
    
    print(f"\n  Z-Score Signal: {z_signal}")
    if results['zscore']['traded']:
        print(f"    → Opened position #{results['zscore']['position']['id']}")
    elif z_signal in ['LONG_DISPERSION', 'SHORT_DISPERSION']:
        print(f"    → Max positions reached, trade skipped")
    else:
        print(f"    → No trade (signal below threshold)")
    
    short_prob = ml_prediction.get('short_probability')
    strategy_used = ml_prediction.get('strategy_used', 'N/A')
    if short_prob is not None:
        print(f"\n  ML Signal: {ml_signal} (SHORT prob: {short_prob:.1%}, via {strategy_used})")
    else:
        print(f"\n  ML Signal: {ml_signal}")
    if results['ml']['traded']:
        print(f"    → Opened position #{results['ml']['position']['id']}")
    elif ml_signal in ['SHORT_DISPERSION', 'LONG_DISPERSION']:
        print(f"    → Max positions reached, trade skipped")
    else:
        print(f"    → No trade (no signal triggered)")
    
    # Print portfolio comparison
    manager.print_comparison()
    
    return manager


# Legacy function for backward compatibility
def run_trading_session_legacy(signal):
    """Legacy trading session (Z-score only)."""
    trader = DispersionPaperTrader(strategy_name="zscore")
    
    print("\n" + "=" * 60)
    print(f"TRADING SESSION - {INDEX_SYMBOL}")
    print("=" * 60)
    
    trader.print_portfolio_summary()
    
    positions_to_close = trader.check_exit_conditions()
    
    if positions_to_close:
        print(f"\n{len(positions_to_close)} position(s) ready to close")
        for position in positions_to_close:
            exit_corr = signal.get('impl_corr')
            exit_z = signal.get('z_score')
            trader.close_position(position, exit_corr, exit_z)
    
    if signal['signal'] in ['LONG_DISPERSION', 'SHORT_DISPERSION']:
        print(f"\nActionable signal: {signal['signal']}")
        trader.execute_paper_trade(signal)
    else:
        print(f"\nNo actionable signal: {signal['signal']}")
    
    trader.print_portfolio_summary()
    
    return trader


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Test the paper trader."""
    print("=" * 60)
    print(f"PAPER TRADER TEST - {INDEX_SYMBOL}")
    print("=" * 60)
    
    # Create sample signals
    sample_signal = {
        'signal': 'SHORT_DISPERSION',
        'impl_corr': 0.45,
        'z_score': 1.8,
        'timestamp': datetime.now().isoformat(),
        'reason': 'Z-score above threshold'
    }
    
    sample_ml = {
        'ml_signal': 'SHORT_DISPERSION',
        'ml_probability': 0.78
    }
    
    run_trading_session(sample_signal, sample_ml)


if __name__ == "__main__":
    main()
