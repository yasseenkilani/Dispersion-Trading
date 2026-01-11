"""
Intraday Paper Trader
=====================

Tracks positions opened via --log-data mode separately from the main portfolios.
Used to analyze whether entry timing affects profitability.

UPDATED: Now tracks 3 parallel portfolios with VEGA-WEIGHTED sizing:
- Fixed 2% (Z-score baseline) + Vega-weighted allocation
- Kelly 0.5x + Confidence Scaling + Vega-weighted (ML)
- Kelly 1.0x + Confidence Scaling + Vega-weighted (ML)

This is for RESEARCH purposes only - to compare:
- Morning entries vs. afternoon entries
- Multiple entries per day vs. single entry
- Different position sizing strategies
- Vega-neutral vs non-vega-neutral performance

Positions are held for 5 trading days, same as the main strategy.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Import risk manager for Kelly calculations
try:
    from risk_manager import RiskManager
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False

# Import vega-weighted sizer
try:
    from vega_weighted_sizer import VegaWeightedKellySizer, MultiStrategyVegaSizer
    VEGA_SIZER_AVAILABLE = True
except ImportError:
    VEGA_SIZER_AVAILABLE = False

# Configuration
INTRADAY_DIR = "intraday_data"
INTRADAY_POSITIONS_FILE = os.path.join(INTRADAY_DIR, "intraday_positions.json")
INTRADAY_TRADES_FILE = os.path.join(INTRADAY_DIR, "intraday_trades.csv")
HOLDING_PERIOD_DAYS = 5
BASE_CAPITAL = 100000.0
BASE_POSITION_PCT = 0.02  # 2% for fixed strategy
BASE_WIN_RATE = 0.669     # Historical win rate
VEGA_MULTIPLIER = 0.15    # Same as main strategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MultiPortfolioIntradayTrader:
    """
    Intraday paper trader tracking 3 parallel portfolios with vega-weighted sizing.
    
    Portfolios:
    1. fixed: Z-Score signal with fixed 2% + vega-weighted allocation
    2. kelly_0.5x: ML signal with half Kelly + confidence + vega-weighted
    3. kelly_1.0x: ML signal with full Kelly + confidence + vega-weighted
    """
    
    STRATEGIES = ['fixed', 'kelly_0.5x', 'kelly_1.0x']
    
    def __init__(self, use_vega_weighting: bool = True):
        """
        Initialize the multi-portfolio intraday trader.
        
        Args:
            use_vega_weighting: Whether to use vega-weighted position sizing
        """
        self.logger = logging.getLogger(__name__)
        self.use_vega_weighting = use_vega_weighting and VEGA_SIZER_AVAILABLE
        os.makedirs(INTRADAY_DIR, exist_ok=True)
        
        # Initialize vega-weighted sizers if available
        if self.use_vega_weighting:
            self.vega_sizer = MultiStrategyVegaSizer(base_capital=BASE_CAPITAL)
            self.logger.info("[INTRADAY] Vega-weighted sizing ENABLED")
        else:
            self.vega_sizer = None
            self.logger.info("[INTRADAY] Vega-weighted sizing DISABLED")
        
        # Initialize risk managers for Kelly calculations (fallback)
        if RISK_MANAGER_AVAILABLE:
            self.risk_managers = {
                'fixed': RiskManager(
                    base_capital=BASE_CAPITAL,
                    kelly_fraction=0,
                    base_position_pct=BASE_POSITION_PCT
                ),
                'kelly_0.5x': RiskManager(
                    base_capital=BASE_CAPITAL,
                    kelly_fraction=0.5,
                    base_win_rate=BASE_WIN_RATE
                ),
                'kelly_1.0x': RiskManager(
                    base_capital=BASE_CAPITAL,
                    kelly_fraction=1.0,
                    base_win_rate=BASE_WIN_RATE
                )
            }
        else:
            self.risk_managers = None
        
        # Load portfolios
        self.portfolios = self._load_portfolios()
    
    def _get_portfolio_file(self) -> str:
        """Get the file path for intraday portfolios."""
        return os.path.join(INTRADAY_DIR, "intraday_multi_portfolios.json")
    
    def _load_portfolios(self) -> Dict:
        """Load all portfolios from file."""
        filepath = self._get_portfolio_file()
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.logger.info(f"[INTRADAY] Loaded portfolios")
                    return data
            except Exception as e:
                self.logger.error(f"[INTRADAY] Error loading portfolios: {e}")
        
        # Initialize new portfolios
        return {
            strategy: {
                'capital': BASE_CAPITAL,
                'positions': [],
                'trade_history': []
            }
            for strategy in self.STRATEGIES
        }
    
    def _save_portfolios(self):
        """Save all portfolios to file."""
        filepath = self._get_portfolio_file()
        
        with open(filepath, 'w') as f:
            json.dump(self.portfolios, f, indent=2, default=str)
    
    def _count_trading_days(self, start_date: datetime, end_date: datetime) -> int:
        """Count trading days between two dates (excludes weekends)."""
        trading_days = 0
        current = start_date
        
        while current < end_date:
            if current.weekday() < 5:  # Monday to Friday
                trading_days += 1
            current += timedelta(days=1)
        
        return trading_days
    
    def calculate_vega_weighted_positions(self, 
                                           signal_type: str,
                                           ml_probability: float,
                                           use_ibkr: bool = False) -> Dict[str, Dict]:
        """
        Calculate vega-weighted position sizes for all strategies.
        
        Args:
            signal_type: 'SHORT_DISPERSION' or 'LONG_DISPERSION'
            ml_probability: ML model's probability
            use_ibkr: Whether to use live IBKR data for Greeks
            
        Returns:
            Dict of strategy -> vega-weighted position info
        """
        if not self.use_vega_weighting or not self.vega_sizer:
            return self._calculate_simple_positions(ml_probability)
        
        # Get current capitals
        current_capitals = {
            strategy: self.portfolios[strategy]['capital']
            for strategy in self.STRATEGIES
        }
        
        # Calculate vega-weighted positions for all strategies
        return self.vega_sizer.calculate_all_positions(
            signal_type=signal_type,
            current_capitals=current_capitals,
            ml_probability=ml_probability,
            use_ibkr=use_ibkr
        )
    
    def _calculate_simple_positions(self, ml_probability: float) -> Dict[str, Dict]:
        """Fallback to simple Kelly-based sizing without vega weighting."""
        results = {}
        
        for strategy in self.STRATEGIES:
            capital = self.portfolios[strategy]['capital']
            
            if self.risk_managers and strategy in self.risk_managers:
                rm = self.risk_managers[strategy]
                use_kelly = strategy != 'fixed'
                
                size_info = rm.get_position_size(
                    current_capital=capital,
                    ml_probability=ml_probability,
                    use_kelly=use_kelly
                )
            else:
                # Fallback if risk manager not available
                if strategy == 'fixed':
                    position_pct = BASE_POSITION_PCT
                elif strategy == 'kelly_0.5x':
                    position_pct = min(0.169, 0.20)
                else:
                    position_pct = min(0.338, 0.20)
                
                size_info = {
                    'position_size': capital * position_pct,
                    'position_pct': position_pct,
                    'kelly_pct': position_pct,
                    'confidence_multiplier': 1.0,
                    'size_reduction': 1.0,
                    'can_trade': True
                }
            
            size_info['strategy'] = strategy
            size_info['capital'] = capital
            size_info['total_position'] = size_info.get('position_size', 0)
            size_info['is_vega_weighted'] = False
            results[strategy] = size_info
        
        return results
    
    def open_positions(self, signal: Dict, ml_prediction: Dict, 
                       use_ibkr: bool = False) -> Dict[str, Optional[Dict]]:
        """
        Open positions for all strategies based on their signals.
        
        Args:
            signal: Z-score based signal dict
            ml_prediction: ML prediction dict
            use_ibkr: Whether to use live IBKR data for vega calculations
            
        Returns:
            Dict of strategy -> position (or None if not opened)
        """
        results = {}
        ml_prob = ml_prediction.get('short_probability', 0.5)
        
        # Determine signal type for vega weighting
        z_signal = signal.get('signal', 'NO_TRADE')
        ml_signal = ml_prediction.get('ml_signal', 'NO_TRADE')
        
        # Get the primary signal type (prefer ML for Kelly strategies)
        if ml_signal in ['SHORT_DISPERSION', 'LONG_DISPERSION']:
            primary_signal_type = ml_signal
        elif z_signal in ['SHORT_DISPERSION', 'LONG_DISPERSION']:
            primary_signal_type = z_signal
        else:
            primary_signal_type = 'NO_TRADE'
        
        # Calculate vega-weighted positions if we have a signal
        if primary_signal_type != 'NO_TRADE' and self.use_vega_weighting:
            vega_positions = self.calculate_vega_weighted_positions(
                signal_type=primary_signal_type,
                ml_probability=ml_prob,
                use_ibkr=use_ibkr
            )
        else:
            vega_positions = self._calculate_simple_positions(ml_prob)
        
        timestamp = datetime.now()
        
        for strategy in self.STRATEGIES:
            results[strategy] = None
            
            # Determine which signal to use for this strategy
            if strategy == 'fixed':
                trade_signal = z_signal
                signal_source = 'Z-Score'
            else:
                trade_signal = ml_signal
                signal_source = 'ML'
            
            # Check if signal is actionable
            if trade_signal not in ['SHORT_DISPERSION', 'LONG_DISPERSION']:
                continue
            
            # Get position sizing info
            size_info = vega_positions.get(strategy, {})
            
            if not size_info.get('can_trade', True):
                self.logger.info(f"[INTRADAY-{strategy.upper()}] Trade blocked: {size_info.get('reason', 'Unknown')}")
                continue
            
            # Extract position details
            total_position = size_info.get('total_position', 0)
            if total_position <= 0:
                continue
            
            # Get vega info if available
            is_vega_weighted = 'index' in size_info and 'components' in size_info
            vega_summary = size_info.get('summary', {}) if is_vega_weighted else {}
            
            # Create position
            position = {
                'id': len(self.portfolios[strategy]['positions']) + 1,
                'entry_timestamp': timestamp.isoformat(),
                'entry_date': timestamp.strftime('%Y-%m-%d'),
                'entry_time': timestamp.strftime('%H:%M:%S'),
                'exit_timestamp': None,
                'signal_type': trade_signal,
                'signal_source': signal_source,
                'entry_corr': signal.get('impl_corr'),
                'entry_z_score': signal.get('z_score'),
                'entry_ml_prob': ml_prob,
                'exit_corr': None,
                'position_size': total_position,
                'position_pct': total_position / self.portfolios[strategy]['capital'],
                'kelly_pct': size_info.get('kelly_sizing', {}).get('kelly_pct', 0),
                'confidence_mult': size_info.get('kelly_sizing', {}).get('confidence_multiplier', 1.0),
                'is_vega_weighted': is_vega_weighted,
                'vega_neutral': vega_summary.get('is_vega_neutral', False),
                'vega_ratio': vega_summary.get('vega_ratio', 1.0),
                'index_allocation': size_info.get('index', {}).get('allocation', 0) if is_vega_weighted else 0,
                'component_allocation': vega_summary.get('component_allocation', 0),
                'status': 'OPEN',
                'pnl': None,
                'holding_period': HOLDING_PERIOD_DAYS
            }
            
            self.portfolios[strategy]['positions'].append(position)
            results[strategy] = position
            
            vega_str = "✅ Vega-neutral" if position['vega_neutral'] else "⚠️ Not vega-neutral"
            self.logger.info(f"[INTRADAY-{strategy.upper()}] Opened position #{position['id']}: "
                           f"{trade_signal} (${total_position:,.0f}, {position['position_pct']:.1%}) {vega_str}")
        
        self._save_portfolios()
        return results
    
    def check_and_close_positions(self, current_corr: float = None) -> Dict[str, List[Dict]]:
        """
        Check and close expired positions for all strategies.
        
        Returns:
            Dict of strategy -> list of closed positions
        """
        results = {strategy: [] for strategy in self.STRATEGIES}
        
        for strategy in self.STRATEGIES:
            for position in self.portfolios[strategy]['positions']:
                if position['status'] != 'OPEN':
                    continue
                
                entry_ts = datetime.fromisoformat(position['entry_timestamp'])
                trading_days = self._count_trading_days(entry_ts, datetime.now())
                
                if trading_days >= position['holding_period']:
                    # Close position
                    position['exit_timestamp'] = datetime.now().isoformat()
                    position['exit_corr'] = current_corr
                    position['status'] = 'CLOSED'
                    
                    # Calculate P&L
                    if position['entry_corr'] is not None and current_corr is not None:
                        corr_change = current_corr - position['entry_corr']
                        
                        if position['signal_type'] == 'SHORT_DISPERSION':
                            pnl = -corr_change * position['position_size'] * VEGA_MULTIPLIER
                        else:
                            pnl = corr_change * position['position_size'] * VEGA_MULTIPLIER
                        
                        position['pnl'] = pnl
                        self.portfolios[strategy]['capital'] += pnl
                    else:
                        position['pnl'] = 0
                    
                    # Add to trade history
                    self.portfolios[strategy]['trade_history'].append({
                        'id': position['id'],
                        'entry_timestamp': position['entry_timestamp'],
                        'exit_timestamp': position['exit_timestamp'],
                        'signal_type': position['signal_type'],
                        'position_size': position['position_size'],
                        'position_pct': position['position_pct'],
                        'is_vega_weighted': position.get('is_vega_weighted', False),
                        'vega_neutral': position.get('vega_neutral', False),
                        'entry_corr': position['entry_corr'],
                        'exit_corr': current_corr,
                        'pnl': position['pnl'],
                        'is_win': position['pnl'] > 0 if position['pnl'] else False
                    })
                    
                    results[strategy].append(position)
                    self.logger.info(f"[INTRADAY-{strategy.upper()}] Closed position #{position['id']}: "
                                   f"P&L ${position['pnl']:,.2f}")
        
        self._save_portfolios()
        return results
    
    def get_stats(self, strategy: str) -> Dict:
        """Get statistics for a strategy."""
        portfolio = self.portfolios[strategy]
        open_positions = [p for p in portfolio['positions'] if p['status'] == 'OPEN']
        closed_trades = portfolio['trade_history']
        
        total_pnl = sum(t.get('pnl', 0) or 0 for t in closed_trades)
        wins = sum(1 for t in closed_trades if t.get('is_win', False))
        win_rate = wins / len(closed_trades) * 100 if closed_trades else 0
        
        # Count vega-neutral trades
        vega_neutral_count = sum(1 for t in closed_trades if t.get('vega_neutral', False))
        
        return {
            'strategy': strategy,
            'capital': portfolio['capital'],
            'return_pct': (portfolio['capital'] - BASE_CAPITAL) / BASE_CAPITAL * 100,
            'open_positions': len(open_positions),
            'closed_trades': len(closed_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'vega_neutral_trades': vega_neutral_count
        }
    
    def print_summary(self):
        """Print summary of all intraday portfolios."""
        print("\n" + "=" * 100)
        print("📊 INTRADAY MULTI-PORTFOLIO TRACKING (VEGA-WEIGHTED)")
        print("=" * 100)
        
        # Portfolio comparison
        stats = {s: self.get_stats(s) for s in self.STRATEGIES}
        
        print(f"\n{'Strategy':<15} {'Capital':>15} {'Return':>10} {'Open':>8} {'Closed':>8} {'Win Rate':>10} {'Vega-Neutral':>12}")
        print("-" * 85)
        
        for strategy in self.STRATEGIES:
            s = stats[strategy]
            vn_str = f"{s['vega_neutral_trades']}/{s['closed_trades']}" if s['closed_trades'] > 0 else "0/0"
            print(f"{strategy:<15} ${s['capital']:>13,.0f} {s['return_pct']:>9.2f}% {s['open_positions']:>8} "
                  f"{s['closed_trades']:>8} {s['win_rate']:>9.1f}% {vn_str:>12}")
        
        # Open positions
        print("\n" + "-" * 100)
        print("OPEN POSITIONS")
        print("-" * 100)
        
        for strategy in self.STRATEGIES:
            open_pos = [p for p in self.portfolios[strategy]['positions'] if p['status'] == 'OPEN']
            
            if open_pos:
                print(f"\n  [{strategy.upper()}]")
                for pos in open_pos:
                    entry_ts = datetime.fromisoformat(pos['entry_timestamp'])
                    trading_days = self._count_trading_days(entry_ts, datetime.now())
                    remaining = pos['holding_period'] - trading_days
                    vega_str = "✅VN" if pos.get('vega_neutral', False) else "⚠️"
                    
                    print(f"    #{pos['id']}: {pos['entry_date']} {pos['entry_time'][:5]} | "
                          f"{pos['signal_type'][:5]} | Size: ${pos['position_size']:,.0f} ({pos['position_pct']:.1%}) | "
                          f"Days: {trading_days}/5 | {vega_str}")
        
        # Recent closed trades
        all_closed = []
        for strategy in self.STRATEGIES:
            for trade in self.portfolios[strategy]['trade_history'][-3:]:
                trade['strategy'] = strategy
                all_closed.append(trade)
        
        if all_closed:
            print("\n" + "-" * 100)
            print("RECENT CLOSED TRADES")
            print("-" * 100)
            
            for trade in sorted(all_closed, key=lambda x: x.get('exit_timestamp', ''), reverse=True)[:5]:
                pnl = trade.get('pnl', 0) or 0
                pnl_str = f"+${pnl:.0f}" if pnl >= 0 else f"-${abs(pnl):.0f}"
                status = "✅" if pnl > 0 else "❌"
                vega_str = "VN" if trade.get('vega_neutral', False) else "--"
                print(f"  [{trade['strategy']:<12}] {trade['signal_type'][:5]} | "
                      f"Size: ${trade['position_size']:,.0f} | P&L: {pnl_str:>8} {status} | {vega_str}")
        
        print("=" * 100)


def run_intraday_tracking(signal: Dict, ml_prediction: Dict, 
                          current_corr: float = None,
                          use_vega_weighting: bool = True,
                          use_ibkr: bool = False):
    """
    Main function to run intraday tracking with all 3 portfolios.
    
    Called from run_daily.py when --log-data is used.
    
    Args:
        signal: Z-score based signal dict
        ml_prediction: ML prediction dict
        current_corr: Current implied correlation (for closing positions)
        use_vega_weighting: Whether to use vega-weighted sizing
        use_ibkr: Whether to use live IBKR data for Greeks
    """
    trader = MultiPortfolioIntradayTrader(use_vega_weighting=use_vega_weighting)
    
    # Get current correlation
    if current_corr is None:
        current_corr = signal.get('impl_corr')
    
    # Check and close expired positions
    closed = trader.check_and_close_positions(current_corr)
    total_closed = sum(len(c) for c in closed.values())
    
    if total_closed > 0:
        print(f"\n✅ Closed {total_closed} intraday position(s)")
        for strategy, positions in closed.items():
            for pos in positions:
                pnl = pos.get('pnl', 0) or 0
                pnl_str = f"+${pnl:.0f}" if pnl >= 0 else f"-${abs(pnl):.0f}"
                vega_str = "VN" if pos.get('vega_neutral', False) else "--"
                print(f"   [{strategy}] #{pos['id']}: {pos['signal_type'][:5]} | P&L: {pnl_str} | {vega_str}")
    
    # Open new positions with vega weighting
    opened = trader.open_positions(signal, ml_prediction, use_ibkr=use_ibkr)
    total_opened = sum(1 for p in opened.values() if p is not None)
    
    if total_opened > 0:
        print(f"\n📈 Opened {total_opened} intraday position(s) {'(VEGA-WEIGHTED)' if use_vega_weighting else ''}")
        for strategy, pos in opened.items():
            if pos:
                vega_str = "✅ Vega-neutral" if pos.get('vega_neutral', False) else "⚠️ Not vega-neutral"
                print(f"   [{strategy}] #{pos['id']}: {pos['signal_type'][:5]} | "
                      f"${pos['position_size']:,.0f} ({pos['position_pct']:.1%}) | {vega_str}")
    
    # Print summary
    trader.print_summary()
    
    return trader
