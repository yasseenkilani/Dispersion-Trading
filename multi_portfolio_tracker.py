"""
Multi-Portfolio Tracker for Dispersion Trading System

Tracks 3 parallel portfolios with different position sizing strategies:
1. Z-Score Fixed (2% per trade) - Baseline
2. ML with 0.5x Kelly + Confidence Scaling
3. ML with 1.0x Kelly + Confidence Scaling

All strategies use VEGA-WEIGHTED position sizing by default.

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

# Try to import vega-weighted sizer
try:
    from vega_weighted_sizer import MultiStrategyVegaSizer, VEGA_SIZER_AVAILABLE
    VEGA_AVAILABLE = VEGA_SIZER_AVAILABLE
except ImportError:
    VEGA_AVAILABLE = False

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
    1. fixed: Z-Score signal with fixed 2% position sizing + vega-weighted
    2. kelly_0.5x: ML signal with half Kelly + confidence scaling + vega-weighted
    3. kelly_1.0x: ML signal with full Kelly + confidence scaling + vega-weighted
    """
    
    STRATEGIES = ['fixed', 'kelly_0.5x', 'kelly_1.0x']
    
    def __init__(self, base_capital: float = INITIAL_CAPITAL, use_vega_weighting: bool = True):
        """
        Initialize the multi-portfolio tracker.
        
        Args:
            base_capital: Starting capital for each portfolio
            use_vega_weighting: Whether to use vega-weighted position sizing
        """
        self.base_capital = base_capital
        self.use_vega_weighting = use_vega_weighting and VEGA_AVAILABLE
        
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
        
        # Initialize vega-weighted sizer if available
        if self.use_vega_weighting:
            self.vega_sizer = MultiStrategyVegaSizer(base_capital=base_capital)
            logger.info("Vega-weighted sizing ENABLED")
        else:
            self.vega_sizer = None
            logger.info("Vega-weighted sizing DISABLED")
        
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
    
    def calculate_position_sizes(self, 
                                  ml_probability: float,
                                  signal_type: str = 'SHORT_DISPERSION',
                                  use_ibkr: bool = False) -> Dict[str, Dict]:
        """
        Calculate position sizes for all strategies.
        
        Args:
            ml_probability: ML model's probability for the trade
            signal_type: 'SHORT_DISPERSION' or 'LONG_DISPERSION'
            use_ibkr: Whether to use live IBKR data for Greeks
            
        Returns:
            Dict of strategy -> position sizing info
        """
        # Get current capitals
        current_capitals = {
            strategy: self.portfolios[strategy]['capital']
            for strategy in self.STRATEGIES
        }
        
        # Use vega-weighted sizing if enabled
        if self.use_vega_weighting and self.vega_sizer:
            return self.vega_sizer.calculate_all_positions(
                signal_type=signal_type,
                current_capitals=current_capitals,
                ml_probability=ml_probability,
                use_ibkr=use_ibkr
            )
        
        # Fallback to simple Kelly-based sizing
        results = {}
        
        for strategy in self.STRATEGIES:
            capital = current_capitals[strategy]
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
            size_info['total_position'] = size_info.get('position_size', 0)
            size_info['is_vega_weighted'] = False
            results[strategy] = size_info
        
        return results
    
    def execute_trades(self, 
                       zscore_signal: Dict,
                       ml_prediction: Dict,
                       use_ibkr: bool = False) -> Dict[str, Dict]:
        """
        Execute trades for all strategies based on their signals.
        
        Args:
            zscore_signal: Z-score based signal dict
            ml_prediction: ML prediction dict with ml_signal and short_probability
            use_ibkr: Whether to use live IBKR data for Greeks
            
        Returns:
            Dict of strategy -> trade result
        """
        results = {}
        
        # Get ML probability for position sizing
        ml_prob = ml_prediction.get('short_probability', 0.5)
        
        # Determine signal type for vega weighting
        z_signal = zscore_signal.get('signal', 'NO_TRADE')
        ml_signal = ml_prediction.get('ml_signal', 'NO_TRADE')
        
        if ml_signal in ['SHORT_DISPERSION', 'LONG_DISPERSION']:
            signal_type = ml_signal
        elif z_signal in ['SHORT_DISPERSION', 'LONG_DISPERSION']:
            signal_type = z_signal
        else:
            signal_type = 'SHORT_DISPERSION'  # Default for sizing calculation
        
        # Calculate position sizes for all strategies
        position_sizes = self.calculate_position_sizes(ml_prob, signal_type, use_ibkr)
        
        # Check and close expired positions first
        self._check_and_close_all_positions(zscore_signal)
        
        # Execute trades for each strategy
        for strategy in self.STRATEGIES:
            size_info = position_sizes[strategy]
            
            results[strategy] = {
                'traded': False,
                'position': None,
                'reason': None,
                'position_size': size_info
            }
            
            # Determine which signal to use
            if strategy == 'fixed':
                # Fixed strategy uses Z-score signal
                signal = z_signal
                signal_source = 'Z-Score'
            else:
                # Kelly strategies use ML signal
                signal = ml_signal
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
            if not size_info.get('can_trade', True):
                results[strategy]['reason'] = size_info.get('reason', 'Risk controls')
                continue
            
            # Get position size
            total_position = size_info.get('total_position', 0)
            if total_position <= 0:
                results[strategy]['reason'] = "Position size is zero"
                continue
            
            position_pct = total_position / self.portfolios[strategy]['capital']
            
            # Get vega info
            is_vega_weighted = 'index' in size_info and 'components' in size_info
            vega_summary = size_info.get('summary', {}) if is_vega_weighted else {}
            
            # Execute the trade
            position = self._open_position(
                strategy=strategy,
                signal_type=signal,
                signal_source=signal_source,
                position_size=total_position,
                position_pct=position_pct,
                zscore_signal=zscore_signal,
                ml_prediction=ml_prediction,
                is_vega_weighted=is_vega_weighted,
                vega_summary=vega_summary
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
                       ml_prediction: Dict,
                       is_vega_weighted: bool = False,
                       vega_summary: Dict = None) -> Dict:
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
            'is_vega_weighted': is_vega_weighted,
            'vega_neutral': vega_summary.get('is_vega_neutral', False) if vega_summary else False,
            'vega_ratio': vega_summary.get('vega_ratio', 1.0) if vega_summary else 1.0,
            'status': 'OPEN',
            'pnl': None,
            'holding_period': HOLDING_PERIOD_DAYS
        }
        
        self.portfolios[strategy]['positions'].append(position)
        
        vega_str = "✅VN" if position['vega_neutral'] else ""
        logger.info(f"[{strategy.upper()}] Opened position #{position['id']}: {signal_type} (${position_size:,.0f}, {position_pct:.1%}) {vega_str}")
        
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
            'is_vega_weighted': position.get('is_vega_weighted', False),
            'vega_neutral': position.get('vega_neutral', False),
            'entry_corr': position['entry_corr'],
            'exit_corr': exit_corr,
            'pnl': position['pnl'],
            'is_win': is_win
        })
        
        pnl_str = f"+${position['pnl']:.0f}" if position['pnl'] >= 0 else f"-${abs(position['pnl']):.0f}"
        vega_str = "VN" if position.get('vega_neutral', False) else "--"
        logger.info(f"[{strategy.upper()}] Closed position #{position['id']}: {position['signal_type']} | P&L: {pnl_str} | {vega_str}")
    
    def get_stats(self, strategy: str) -> Dict:
        """Get statistics for a strategy."""
        portfolio = self.portfolios[strategy]
        open_positions = self.get_open_positions(strategy)
        closed_trades = portfolio['trade_history']
        
        total_pnl = sum(t.get('pnl', 0) or 0 for t in closed_trades)
        wins = sum(1 for t in closed_trades if t.get('is_win', False))
        win_rate = wins / len(closed_trades) * 100 if closed_trades else 0
        
        # Count vega-neutral trades
        vega_neutral_count = sum(1 for t in closed_trades if t.get('vega_neutral', False))
        
        return {
            'strategy': strategy,
            'capital': portfolio['capital'],
            'return_pct': (portfolio['capital'] - self.base_capital) / self.base_capital * 100,
            'open_positions': len(open_positions),
            'closed_trades': len(closed_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'vega_neutral_trades': vega_neutral_count
        }
    
    def print_comparison(self):
        """Print comparison of all portfolios."""
        print("\n" + "=" * 90)
        print("PORTFOLIO COMPARISON")
        print("=" * 90)
        
        stats = {s: self.get_stats(s) for s in self.STRATEGIES}
        
        print(f"\n  {'Strategy':<15} {'Capital':>15} {'Return':>10} {'Open':>8} {'Closed':>8} {'Win Rate':>10} {'Vega-Neutral':>12}")
        print("  " + "-" * 85)
        
        for strategy in self.STRATEGIES:
            s = stats[strategy]
            vn_str = f"{s['vega_neutral_trades']}/{s['closed_trades']}" if s['closed_trades'] > 0 else "0/0"
            print(f"  {strategy:<15} ${s['capital']:>13,.0f} {s['return_pct']:>9.2f}% {s['open_positions']:>8} "
                  f"{s['closed_trades']:>8} {s['win_rate']:>9.1f}% {vn_str:>12}")
    
    def print_open_positions(self):
        """Print all open positions."""
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
                    vega_str = "✅VN" if pos.get('vega_neutral', False) else "⚠️"
                    
                    print(f"    #{pos['id']}: {pos['signal_type'][:5]} | "
                          f"Size: ${pos['position_size']:,.0f} ({pos['position_pct']:.1%}) | "
                          f"Days: {trading_days}/{pos['holding_period']} | {vega_str}")
            else:
                print(f"\n  [{strategy.upper()}] No open positions")
    
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

def run_multi_portfolio_session(zscore_signal: Dict, 
                                 ml_prediction: Dict,
                                 use_vega_weighting: bool = True,
                                 use_ibkr: bool = False) -> MultiPortfolioTrader:
    """
    Run a complete trading session with all 3 strategies.
    
    Args:
        zscore_signal: Z-score based signal
        ml_prediction: ML prediction dict
        use_vega_weighting: Whether to use vega-weighted position sizing
        use_ibkr: Whether to use live IBKR data for Greeks
        
    Returns:
        MultiPortfolioTrader instance
    """
    trader = MultiPortfolioTrader(use_vega_weighting=use_vega_weighting)
    
    print("\n" + "=" * 90)
    print(f"MULTI-PORTFOLIO TRADING SESSION - {INDEX_SYMBOL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Vega-Weighted: {'✅ ENABLED' if trader.use_vega_weighting else '❌ DISABLED'}")
    print(f"Greeks Source: {'LIVE (IBKR)' if use_ibkr else 'SIMULATED'}")
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
    
    # Determine signal type for sizing
    if ml_signal in ['SHORT_DISPERSION', 'LONG_DISPERSION']:
        signal_type = ml_signal
    elif z_signal in ['SHORT_DISPERSION', 'LONG_DISPERSION']:
        signal_type = z_signal
    else:
        signal_type = 'SHORT_DISPERSION'
    
    # Calculate and display position sizes
    print("\n" + "-" * 90)
    print("POSITION SIZING" + (" (VEGA-WEIGHTED)" if trader.use_vega_weighting else ""))
    print("-" * 90)
    
    position_sizes = trader.calculate_position_sizes(ml_prob, signal_type, use_ibkr)
    
    if trader.use_vega_weighting:
        # Vega-weighted display
        print(f"\n  {'Strategy':<15} {'Total Pos':>12} {'Index':>12} {'Components':>12} {'Vega-Neutral':>14}")
        print("  " + "-" * 70)
        
        for strategy in trader.STRATEGIES:
            ps = position_sizes[strategy]
            total = ps.get('total_position', 0)
            index_alloc = ps.get('index', {}).get('allocation', 0) if 'index' in ps else total / 2
            comp_alloc = ps.get('summary', {}).get('component_allocation', 0) if 'summary' in ps else total / 2
            is_vn = ps.get('summary', {}).get('is_vega_neutral', False) if 'summary' in ps else False
            vn_str = "✅ Yes" if is_vn else "⚠️ No"
            
            print(f"  {strategy:<15} ${total:>10,.0f} ${index_alloc:>10,.0f} ${comp_alloc:>10,.0f} {vn_str:>14}")
    else:
        # Simple Kelly display
        print(f"\n  {'Strategy':<15} {'Kelly %':>10} {'Conf Mult':>10} {'Size Red':>10} {'Final %':>10} {'Amount':>15}")
        print("  " + "-" * 75)
        
        for strategy in trader.STRATEGIES:
            ps = position_sizes[strategy]
            kelly_pct = ps.get('kelly_pct', ps.get('kelly_sizing', {}).get('kelly_pct', 0))
            conf_mult = ps.get('confidence_multiplier', ps.get('kelly_sizing', {}).get('confidence_multiplier', 1.0))
            size_red = ps.get('size_reduction', ps.get('kelly_sizing', {}).get('size_reduction', 1.0))
            pos_pct = ps.get('position_pct', ps.get('total_position', 0) / trader.portfolios[strategy]['capital'])
            pos_size = ps.get('total_position', ps.get('position_size', 0))
            
            kelly_str = f"{kelly_pct:.1%}" if kelly_pct > 0 else "N/A"
            print(f"  {strategy:<15} {kelly_str:>10} {conf_mult:>10.2f}x {size_red:>9.0%} {pos_pct:>10.1%} ${pos_size:>13,.0f}")
    
    # Execute trades
    results = trader.execute_trades(zscore_signal, ml_prediction, use_ibkr)
    
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
            vega_str = "✅VN" if pos.get('vega_neutral', False) else ""
            print(f"    Position #{pos['id']}: {pos['signal_type']} | ${pos['position_size']:,.0f} ({pos['position_pct']:.1%}) {vega_str}")
    
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
    
    # Test with vega-weighted (default)
    run_multi_portfolio_session(sample_zscore, sample_ml, use_vega_weighting=True, use_ibkr=False)


if __name__ == "__main__":
    main()
