"""
Multi-Portfolio Tracker for Dispersion Trading System

Tracks 3 parallel portfolios with different position sizing strategies:
1. Z-Score Fixed (2% per trade) - Baseline
2. ML with 0.5x Kelly + Confidence Scaling
3. ML with 1.0x Kelly + Confidence Scaling

This allows direct comparison of risk management approaches during paper trading.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from risk_manager import RiskManager, MultiStrategyRiskManager
from ibkr_connector import NDX_COMPONENTS, INDEX_SYMBOL

# =============================================================================
# CONFIGURATION
# =============================================================================

INITIAL_CAPITAL = 100000
MAX_POSITIONS = 3
HOLDING_PERIOD_DAYS = 5
NUM_COMPONENTS_TO_TRADE = 30

# Position sizing
BASE_POSITION_PCT = 0.02  # 2% for fixed strategy
BASE_WIN_RATE = 0.669     # Historical win rate from backtest

# Paths
POSITIONS_DIR = "positions"
LOG_DIR = "logs"

logger = logging.getLogger(__name__)


# =============================================================================
# MULTI-PORTFOLIO PAPER TRADER
# =============================================================================

class MultiPortfolioTrader:
    """
    Tracks 3 parallel portfolios with different position sizing strategies.
    
    Portfolios:
    1. fixed: Z-Score signal with fixed 2% position sizing
    2. kelly_0.5x: ML signal with half Kelly + confidence scaling
    3. kelly_1.0x: ML signal with full Kelly + confidence scaling
    """
    
    STRATEGIES = ['fixed', 'kelly_0.5x', 'kelly_1.0x']
    
    def __init__(self, base_capital: float = INITIAL_CAPITAL):
        """Initialize the multi-portfolio tracker."""
        self.base_capital = base_capital
        
        # Create directories
        os.makedirs(POSITIONS_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Initialize risk managers for each strategy
        self.risk_managers = {
            'fixed': RiskManager(
                base_capital=base_capital,
                kelly_fraction=0,  # Not used
                base_position_pct=BASE_POSITION_PCT,
                data_dir=os.path.dirname(os.path.abspath(__file__))
            ),
            'kelly_0.5x': RiskManager(
                base_capital=base_capital,
                kelly_fraction=0.5,
                base_win_rate=BASE_WIN_RATE,
                data_dir=os.path.dirname(os.path.abspath(__file__))
            ),
            'kelly_1.0x': RiskManager(
                base_capital=base_capital,
                kelly_fraction=1.0,
                base_win_rate=BASE_WIN_RATE,
                data_dir=os.path.dirname(os.path.abspath(__file__))
            )
        }
        
        # Override file paths to keep separate
        for name, rm in self.risk_managers.items():
            rm.risk_data_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                POSITIONS_DIR,
                f'risk_data_{name}.json'
            )
            rm.risk_data = rm._load_risk_data()
        
        # Portfolio data
        self.portfolios = {name: self._load_portfolio(name) for name in self.STRATEGIES}
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging."""
        log_file = os.path.join(LOG_DIR, f"multi_portfolio_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _get_portfolio_file(self, strategy: str) -> str:
        """Get the file path for a strategy's portfolio."""
        return os.path.join(POSITIONS_DIR, f"{strategy}_portfolio.json")
    
    def _load_portfolio(self, strategy: str) -> Dict:
        """Load portfolio data from file."""
        filepath = self._get_portfolio_file(strategy)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load {strategy} portfolio: {e}")
        
        # Initialize new portfolio
        return {
            'strategy': strategy,
            'capital': self.base_capital,
            'positions': [],
            'trade_history': [],
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_portfolio(self, strategy: str):
        """Save portfolio data to file."""
        filepath = self._get_portfolio_file(strategy)
        self.portfolios[strategy]['last_updated'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(self.portfolios[strategy], f, indent=2, default=str)
    
    def _save_all_portfolios(self):
        """Save all portfolios."""
        for strategy in self.STRATEGIES:
            self._save_portfolio(strategy)
    
    def _count_trading_days(self, start_date: datetime, end_date: datetime) -> int:
        """Count trading days between two dates (excludes weekends)."""
        trading_days = 0
        current = start_date
        
        while current < end_date:
            if current.weekday() < 5:  # Monday to Friday
                trading_days += 1
            current += timedelta(days=1)
        
        return trading_days
    
    def get_open_positions(self, strategy: str) -> List[Dict]:
        """Get open positions for a strategy."""
        return [p for p in self.portfolios[strategy]['positions'] if p['status'] == 'OPEN']
    
    def can_open_position(self, strategy: str) -> bool:
        """Check if we can open a new position for a strategy."""
        return len(self.get_open_positions(strategy)) < MAX_POSITIONS
    
    def calculate_position_sizes(self, ml_probability: float) -> Dict[str, Dict]:
        """
        Calculate position sizes for all strategies.
        
        Args:
            ml_probability: ML model's probability for the trade
            
        Returns:
            Dict of strategy -> position sizing info
        """
        results = {}
        
        for strategy in self.STRATEGIES:
            capital = self.portfolios[strategy]['capital']
            rm = self.risk_managers[strategy]
            
            if strategy == 'fixed':
                # Fixed 2% sizing, no Kelly
                size_info = rm.get_position_size(
                    current_capital=capital,
                    ml_probability=ml_probability,
                    use_kelly=False
                )
            else:
                # Kelly + confidence scaling
                size_info = rm.get_position_size(
                    current_capital=capital,
                    ml_probability=ml_probability,
                    use_kelly=True
                )
            
            size_info['strategy'] = strategy
            size_info['capital'] = capital
            results[strategy] = size_info
        
        return results
    
    def execute_trades(self, 
                       zscore_signal: Dict,
                       ml_prediction: Dict) -> Dict[str, Dict]:
        """
        Execute trades for all strategies based on their signals.
        
        Args:
            zscore_signal: Z-score based signal dict
            ml_prediction: ML prediction dict with ml_signal and short_probability
            
        Returns:
            Dict of strategy -> trade result
        """
        results = {}
        
        # Get ML probability for position sizing
        ml_prob = ml_prediction.get('short_probability', 0.5)
        
        # Calculate position sizes for all strategies
        position_sizes = self.calculate_position_sizes(ml_prob)
        
        # Check and close expired positions first
        self._check_and_close_all_positions(zscore_signal)
        
        # Execute trades for each strategy
        for strategy in self.STRATEGIES:
            results[strategy] = {
                'traded': False,
                'position': None,
                'reason': None,
                'position_size': position_sizes[strategy]
            }
            
            # Determine which signal to use
            if strategy == 'fixed':
                # Fixed strategy uses Z-score signal
                signal = zscore_signal.get('signal', 'NO_TRADE')
                signal_source = 'Z-Score'
            else:
                # Kelly strategies use ML signal
                signal = ml_prediction.get('ml_signal', 'NO_TRADE')
                signal_source = 'ML'
            
            # Check if signal is actionable
            if signal not in ['LONG_DISPERSION', 'SHORT_DISPERSION']:
                results[strategy]['reason'] = f"No actionable signal ({signal_source})"
                continue
            
            # Check if we can open position
            if not self.can_open_position(strategy):
                results[strategy]['reason'] = "Max positions reached"
                continue
            
            # Check risk controls
            size_info = position_sizes[strategy]
            if not size_info['can_trade']:
                results[strategy]['reason'] = size_info['reason']
                continue
            
            # Execute the trade
            position = self._open_position(
                strategy=strategy,
                signal_type=signal,
                signal_source=signal_source,
                position_size=size_info['position_size'],
                position_pct=size_info['position_pct'],
                zscore_signal=zscore_signal,
                ml_prediction=ml_prediction
            )
            
            results[strategy]['traded'] = True
            results[strategy]['position'] = position
            results[strategy]['reason'] = "Trade executed"
        
        self._save_all_portfolios()
        return results
    
    def _open_position(self,
                       strategy: str,
                       signal_type: str,
                       signal_source: str,
                       position_size: float,
                       position_pct: float,
                       zscore_signal: Dict,
                       ml_prediction: Dict) -> Dict:
        """Open a new position for a strategy."""
        position = {
            'id': len(self.portfolios[strategy]['positions']) + 1,
            'strategy': strategy,
            'signal_source': signal_source,
            'entry_date': datetime.now().isoformat(),
            'exit_date': None,
            'signal_type': signal_type,
            'entry_corr': zscore_signal.get('impl_corr'),
            'entry_z_score': zscore_signal.get('z_score'),
            'entry_ml_prob': ml_prediction.get('short_probability'),
            'exit_corr': None,
            'exit_z_score': None,
            'position_size': position_size,
            'position_pct': position_pct,
            'status': 'OPEN',
            'pnl': None,
            'holding_period': HOLDING_PERIOD_DAYS
        }
        
        self.portfolios[strategy]['positions'].append(position)
        logger.info(f"[{strategy.upper()}] Opened position #{position['id']}: {signal_type} (${position_size:,.0f}, {position_pct:.1%})")
        
        return position
    
    def _check_and_close_all_positions(self, current_signal: Dict):
        """Check and close expired positions for all strategies."""
        exit_corr = current_signal.get('impl_corr')
        exit_z = current_signal.get('z_score')
        
        for strategy in self.STRATEGIES:
            positions_to_close = []
            
            for position in self.get_open_positions(strategy):
                entry_date = datetime.fromisoformat(position['entry_date'])
                trading_days = self._count_trading_days(entry_date, datetime.now())
                
                if trading_days >= position['holding_period']:
                    positions_to_close.append(position)
            
            for position in positions_to_close:
                self._close_position(strategy, position, exit_corr, exit_z)
    
    def _close_position(self, 
                        strategy: str, 
                        position: Dict, 
                        exit_corr: float, 
                        exit_z: float):
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
        self.portfolios[strategy]['capital'] += position['pnl']
        
        # Record trade result in risk manager
        is_win = position['pnl'] > 0
        self.risk_managers[strategy].record_trade_result(
            pnl=position['pnl'],
            is_win=is_win,
            date=datetime.now().strftime('%Y-%m-%d')
        )
        
        # Add to trade history
        self.portfolios[strategy]['trade_history'].append({
            'id': position['id'],
            'entry_date': position['entry_date'],
            'exit_date': position['exit_date'],
            'signal_type': position['signal_type'],
            'position_size': position['position_size'],
            'position_pct': position['position_pct'],
            'entry_corr': position['entry_corr'],
            'exit_corr': exit_corr,
            'pnl': position['pnl'],
            'is_win': is_win
        })
        
        logger.info(f"[{strategy.upper()}] Closed position #{position['id']}: P&L ${position['pnl']:,.2f}")
    
    def get_portfolio_stats(self, strategy: str) -> Dict:
        """Get statistics for a portfolio."""
        portfolio = self.portfolios[strategy]
        open_positions = self.get_open_positions(strategy)
        closed_trades = portfolio['trade_history']
        
        total_pnl = sum(t.get('pnl', 0) or 0 for t in closed_trades)
        wins = sum(1 for t in closed_trades if t.get('is_win', False))
        win_rate = wins / len(closed_trades) * 100 if closed_trades else 0
        
        return {
            'strategy': strategy,
            'capital': portfolio['capital'],
            'return_pct': (portfolio['capital'] - self.base_capital) / self.base_capital * 100,
            'open_positions': len(open_positions),
            'closed_trades': len(closed_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate
        }
    
    def print_comparison(self):
        """Print side-by-side comparison of all strategies."""
        stats = {s: self.get_portfolio_stats(s) for s in self.STRATEGIES}
        
        print("\n" + "=" * 90)
        print("MULTI-PORTFOLIO COMPARISON")
        print("=" * 90)
        
        # Header
        print(f"\n{'Metric':<20} {'Fixed 2%':>20} {'Kelly 0.5x':>20} {'Kelly 1.0x':>20}")
        print("-" * 90)
        
        # Capital
        print(f"{'Capital':<20} ${stats['fixed']['capital']:>18,.0f} ${stats['kelly_0.5x']['capital']:>18,.0f} ${stats['kelly_1.0x']['capital']:>18,.0f}")
        
        # Return
        print(f"{'Return':<20} {stats['fixed']['return_pct']:>19.2f}% {stats['kelly_0.5x']['return_pct']:>19.2f}% {stats['kelly_1.0x']['return_pct']:>19.2f}%")
        
        # Open positions
        print(f"{'Open Positions':<20} {stats['fixed']['open_positions']:>20} {stats['kelly_0.5x']['open_positions']:>20} {stats['kelly_1.0x']['open_positions']:>20}")
        
        # Closed trades
        print(f"{'Closed Trades':<20} {stats['fixed']['closed_trades']:>20} {stats['kelly_0.5x']['closed_trades']:>20} {stats['kelly_1.0x']['closed_trades']:>20}")
        
        # Total P&L
        print(f"{'Total P&L':<20} ${stats['fixed']['total_pnl']:>18,.0f} ${stats['kelly_0.5x']['total_pnl']:>18,.0f} ${stats['kelly_1.0x']['total_pnl']:>18,.0f}")
        
        # Win rate
        print(f"{'Win Rate':<20} {stats['fixed']['win_rate']:>19.1f}% {stats['kelly_0.5x']['win_rate']:>19.1f}% {stats['kelly_1.0x']['win_rate']:>19.1f}%")
        
        print("-" * 90)
        
        # Determine best performer
        returns = [(s, stats[s]['return_pct']) for s in self.STRATEGIES]
        best = max(returns, key=lambda x: x[1])
        
        if best[1] != 0:
            print(f"\nBest Performer: {best[0]} ({best[1]:+.2f}%)")
        
        print("=" * 90)
    
    def print_open_positions(self):
        """Print all open positions across strategies."""
        print("\n" + "-" * 90)
        print("OPEN POSITIONS")
        print("-" * 90)
        
        for strategy in self.STRATEGIES:
            open_pos = self.get_open_positions(strategy)
            
            if open_pos:
                print(f"\n  [{strategy.upper()}]")
                for pos in open_pos:
                    entry_date = datetime.fromisoformat(pos['entry_date'])
                    trading_days = self._count_trading_days(entry_date, datetime.now())
                    remaining = pos['holding_period'] - trading_days
                    
                    print(f"    #{pos['id']}: {pos['signal_type']} | "
                          f"Size: ${pos['position_size']:,.0f} ({pos['position_pct']:.1%}) | "
                          f"Days: {trading_days}/{pos['holding_period']} | "
                          f"Closes in: {remaining}")
    
    def print_risk_status(self):
        """Print risk status for all strategies."""
        print("\n" + "-" * 90)
        print("RISK STATUS")
        print("-" * 90)
        
        for strategy in self.STRATEGIES:
            capital = self.portfolios[strategy]['capital']
            rm = self.risk_managers[strategy]
            summary = rm.get_risk_summary(capital)
            
            status_icon = "🟢" if summary['trading_status'] == 'ACTIVE' else "🔴"
            
            print(f"\n  [{strategy.upper()}] {status_icon} {summary['trading_status']}")
            print(f"    Drawdown: {summary['drawdown_pct']:.1%} | "
                  f"Daily P&L: ${summary['daily_pnl']:,.0f} | "
                  f"Weekly P&L: ${summary['weekly_pnl']:,.0f}")
            print(f"    Consecutive Losses: {summary['consecutive_losses']} | "
                  f"Size Reduction: {summary['size_reduction']:.0%}")
            
            if summary['status_reason'] != 'Normal':
                print(f"    ⚠️  {summary['status_reason']}")
    
    def print_full_summary(self):
        """Print complete summary of all portfolios."""
        self.print_comparison()
        self.print_open_positions()
        self.print_risk_status()


# =============================================================================
# TRADING SESSION
# =============================================================================

def run_multi_portfolio_session(zscore_signal: Dict, ml_prediction: Dict) -> MultiPortfolioTrader:
    """
    Run a complete trading session with all 3 strategies.
    
    Args:
        zscore_signal: Z-score based signal
        ml_prediction: ML prediction dict
        
    Returns:
        MultiPortfolioTrader instance
    """
    trader = MultiPortfolioTrader()
    
    print("\n" + "=" * 90)
    print(f"MULTI-PORTFOLIO TRADING SESSION - {INDEX_SYMBOL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    
    # Print signal information
    print("\n" + "-" * 90)
    print("SIGNALS")
    print("-" * 90)
    
    z_signal = zscore_signal.get('signal', 'NO_TRADE')
    ml_signal = ml_prediction.get('ml_signal', 'NO_TRADE')
    ml_prob = ml_prediction.get('short_probability', 0)
    
    print(f"\n  Z-Score Signal: {z_signal}")
    print(f"    Implied Correlation: {zscore_signal.get('impl_corr', 'N/A'):.4f}")
    print(f"    Z-Score: {zscore_signal.get('z_score', 'N/A'):.2f}")
    
    print(f"\n  ML Signal: {ml_signal}")
    print(f"    SHORT Probability: {ml_prob:.1%}")
    print(f"    Strategy: {ml_prediction.get('strategy_used', 'N/A')}")
    
    # Calculate and display position sizes
    print("\n" + "-" * 90)
    print("POSITION SIZING")
    print("-" * 90)
    
    position_sizes = trader.calculate_position_sizes(ml_prob)
    
    print(f"\n  {'Strategy':<15} {'Kelly %':>10} {'Conf Mult':>10} {'Size Red':>10} {'Final %':>10} {'Amount':>15}")
    print("  " + "-" * 75)
    
    for strategy in trader.STRATEGIES:
        ps = position_sizes[strategy]
        kelly_str = f"{ps['kelly_pct']:.1%}" if ps['kelly_pct'] > 0 else "N/A"
        print(f"  {strategy:<15} {kelly_str:>10} {ps['confidence_multiplier']:>10.2f}x {ps['size_reduction']:>9.0%} {ps['position_pct']:>10.1%} ${ps['position_size']:>13,.0f}")
    
    # Execute trades
    results = trader.execute_trades(zscore_signal, ml_prediction)
    
    # Print trade execution results
    print("\n" + "-" * 90)
    print("TRADE EXECUTION")
    print("-" * 90)
    
    for strategy in trader.STRATEGIES:
        result = results[strategy]
        status = "✓ EXECUTED" if result['traded'] else "✗ SKIPPED"
        print(f"\n  [{strategy.upper()}] {status}")
        print(f"    Reason: {result['reason']}")
        
        if result['traded']:
            pos = result['position']
            print(f"    Position #{pos['id']}: {pos['signal_type']} | ${pos['position_size']:,.0f} ({pos['position_pct']:.1%})")
    
    # Print full summary
    trader.print_full_summary()
    
    return trader


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Test the multi-portfolio tracker."""
    print("=" * 90)
    print("MULTI-PORTFOLIO TRACKER TEST")
    print("=" * 90)
    
    # Create sample signals
    sample_zscore = {
        'signal': 'SHORT_DISPERSION',
        'impl_corr': 0.45,
        'z_score': 1.8,
        'timestamp': datetime.now().isoformat()
    }
    
    sample_ml = {
        'ml_signal': 'SHORT_DISPERSION',
        'short_probability': 0.78,
        'strategy_used': 'ML_MODEL'
    }
    
    run_multi_portfolio_session(sample_zscore, sample_ml)


if __name__ == "__main__":
    main()
