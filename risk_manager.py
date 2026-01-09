"""
Risk Manager Module for Dispersion Trading System

Implements:
- Kelly criterion position sizing (0.5x and 1.0x)
- Confidence-based dynamic allocation
- Drawdown controls and circuit breakers
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages position sizing and risk controls for the dispersion trading system.
    """
    
    # Default drawdown limits
    DEFAULT_LIMITS = {
        'daily_loss_limit': -0.02,      # -2% daily → halt new trades
        'weekly_loss_limit': -0.05,     # -5% weekly → reduce size 50%
        'monthly_loss_limit': -0.10,    # -10% monthly → halt trading
        'consecutive_loss_limit': 5,     # 5 losses → reduce size 50%
    }
    
    def __init__(self, 
                 base_capital: float = 100000,
                 kelly_fraction: float = 0.5,
                 base_win_rate: float = 0.669,
                 base_position_pct: float = 0.02,
                 data_dir: str = None):
        """
        Initialize the risk manager.
        
        Args:
            base_capital: Starting capital
            kelly_fraction: Fraction of Kelly to use (0.5 = half Kelly)
            base_win_rate: Historical win rate for Kelly calculation
            base_position_pct: Base position size as percentage of capital
            data_dir: Directory to store risk management data
        """
        self.base_capital = base_capital
        self.kelly_fraction = kelly_fraction
        self.base_win_rate = base_win_rate
        self.base_position_pct = base_position_pct
        
        # Set up data directory
        if data_dir is None:
            data_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir
        self.risk_data_file = os.path.join(data_dir, 'positions', 'risk_data.json')
        
        # Load or initialize risk data
        self.risk_data = self._load_risk_data()
        
        # Drawdown limits
        self.limits = self.DEFAULT_LIMITS.copy()
        
    def _load_risk_data(self) -> Dict:
        """Load risk tracking data from file."""
        if os.path.exists(self.risk_data_file):
            try:
                with open(self.risk_data_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load risk data: {e}")
        
        # Initialize default risk data
        return {
            'daily_pnl': {},           # date -> pnl
            'trade_results': [],        # list of win/loss (1/0)
            'current_capital': self.base_capital,
            'peak_capital': self.base_capital,
            'consecutive_losses': 0,
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_risk_data(self):
        """Save risk tracking data to file."""
        try:
            os.makedirs(os.path.dirname(self.risk_data_file), exist_ok=True)
            self.risk_data['last_updated'] = datetime.now().isoformat()
            with open(self.risk_data_file, 'w') as f:
                json.dump(self.risk_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save risk data: {e}")
    
    def calculate_kelly_size(self, 
                             win_rate: float = None,
                             win_loss_ratio: float = 1.0) -> float:
        """
        Calculate Kelly criterion position size.
        
        Kelly % = (Win Rate × Win/Loss Ratio - Loss Rate) / Win/Loss Ratio
        
        For 1:1 payoff: Kelly % = 2 × Win Rate - 1
        
        Args:
            win_rate: Win rate (default: base_win_rate)
            win_loss_ratio: Average win / average loss (default: 1.0)
            
        Returns:
            Kelly position size as decimal (e.g., 0.338 for 33.8%)
        """
        if win_rate is None:
            win_rate = self.base_win_rate
            
        loss_rate = 1 - win_rate
        
        # Kelly formula
        kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # Apply fraction
        kelly_adjusted = kelly * self.kelly_fraction
        
        # Ensure non-negative
        return max(0, kelly_adjusted)
    
    def calculate_confidence_multiplier(self, ml_probability: float) -> float:
        """
        Calculate position size multiplier based on ML confidence.
        
        Args:
            ml_probability: ML model's probability (0.0 to 1.0)
            
        Returns:
            Multiplier for position size (0.5 to 1.5)
        """
        if ml_probability >= 0.85:
            return 1.5  # Very high confidence
        elif ml_probability >= 0.75:
            return 1.25  # High confidence
        elif ml_probability >= 0.65:
            return 1.0  # Medium confidence
        elif ml_probability >= 0.60:
            return 0.75  # Low confidence (at threshold)
        else:
            return 0.5  # Below threshold (shouldn't trade)
    
    def get_position_size(self,
                          current_capital: float,
                          ml_probability: float,
                          use_kelly: bool = True) -> Dict:
        """
        Calculate position size incorporating Kelly and confidence scaling.
        
        Args:
            current_capital: Current portfolio capital
            ml_probability: ML model's probability
            use_kelly: Whether to use Kelly sizing (False = fixed 2%)
            
        Returns:
            Dict with position sizing details
        """
        # Check drawdown controls first
        controls = self.check_drawdown_controls(current_capital)
        
        if controls['halt_trading']:
            return {
                'position_size': 0,
                'position_pct': 0,
                'kelly_pct': 0,
                'confidence_multiplier': 0,
                'size_reduction': 1.0,
                'reason': controls['reason'],
                'can_trade': False
            }
        
        # Base calculation
        if use_kelly:
            kelly_pct = self.calculate_kelly_size()
        else:
            kelly_pct = self.base_position_pct
        
        # Confidence multiplier
        conf_multiplier = self.calculate_confidence_multiplier(ml_probability)
        
        # Apply size reduction from drawdown controls
        size_reduction = controls['size_reduction']
        
        # Final position percentage
        final_pct = kelly_pct * conf_multiplier * size_reduction
        
        # Cap at reasonable maximum (20% of capital)
        final_pct = min(final_pct, 0.20)
        
        # Calculate dollar amount
        position_size = current_capital * final_pct
        
        return {
            'position_size': position_size,
            'position_pct': final_pct,
            'kelly_pct': kelly_pct,
            'confidence_multiplier': conf_multiplier,
            'size_reduction': size_reduction,
            'reason': controls.get('reason', 'Normal sizing'),
            'can_trade': True
        }
    
    def check_drawdown_controls(self, current_capital: float) -> Dict:
        """
        Check all drawdown controls and return trading restrictions.
        
        Args:
            current_capital: Current portfolio capital
            
        Returns:
            Dict with halt_trading, size_reduction, and reason
        """
        result = {
            'halt_trading': False,
            'size_reduction': 1.0,
            'reason': 'Normal'
        }
        
        # Update current capital tracking
        self.risk_data['current_capital'] = current_capital
        if current_capital > self.risk_data.get('peak_capital', current_capital):
            self.risk_data['peak_capital'] = current_capital
        
        # Check monthly drawdown (from peak)
        peak = self.risk_data.get('peak_capital', self.base_capital)
        monthly_dd = (current_capital - peak) / peak
        
        if monthly_dd <= self.limits['monthly_loss_limit']:
            result['halt_trading'] = True
            result['reason'] = f"Monthly drawdown limit hit ({monthly_dd:.1%})"
            return result
        
        # Check daily P&L
        today = datetime.now().strftime('%Y-%m-%d')
        daily_pnl = self.risk_data.get('daily_pnl', {}).get(today, 0)
        daily_pnl_pct = daily_pnl / self.base_capital
        
        if daily_pnl_pct <= self.limits['daily_loss_limit']:
            result['halt_trading'] = True
            result['reason'] = f"Daily loss limit hit ({daily_pnl_pct:.1%})"
            return result
        
        # Check weekly P&L
        weekly_pnl = self._get_weekly_pnl()
        weekly_pnl_pct = weekly_pnl / self.base_capital
        
        if weekly_pnl_pct <= self.limits['weekly_loss_limit']:
            result['size_reduction'] = 0.5
            result['reason'] = f"Weekly loss limit - size reduced 50% ({weekly_pnl_pct:.1%})"
        
        # Check consecutive losses
        consecutive = self.risk_data.get('consecutive_losses', 0)
        if consecutive >= self.limits['consecutive_loss_limit']:
            result['size_reduction'] = min(result['size_reduction'], 0.5)
            result['reason'] = f"Consecutive losses ({consecutive}) - size reduced 50%"
        
        self._save_risk_data()
        return result
    
    def _get_weekly_pnl(self) -> float:
        """Calculate P&L for the current week."""
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        
        weekly_pnl = 0
        for date_str, pnl in self.risk_data.get('daily_pnl', {}).items():
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                if date >= week_start:
                    weekly_pnl += pnl
            except:
                continue
        
        return weekly_pnl
    
    def record_trade_result(self, pnl: float, is_win: bool, date: str = None):
        """
        Record a trade result for tracking.
        
        Args:
            pnl: P&L amount
            is_win: Whether the trade was profitable
            date: Date of the trade (default: today)
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Update daily P&L
        if 'daily_pnl' not in self.risk_data:
            self.risk_data['daily_pnl'] = {}
        
        current_daily = self.risk_data['daily_pnl'].get(date, 0)
        self.risk_data['daily_pnl'][date] = current_daily + pnl
        
        # Update trade results
        self.risk_data['trade_results'].append({
            'date': date,
            'pnl': pnl,
            'is_win': is_win
        })
        
        # Update consecutive losses
        if is_win:
            self.risk_data['consecutive_losses'] = 0
        else:
            self.risk_data['consecutive_losses'] = self.risk_data.get('consecutive_losses', 0) + 1
        
        self._save_risk_data()
    
    def get_rolling_win_rate(self, lookback: int = 20) -> float:
        """
        Calculate rolling win rate from recent trades.
        
        Args:
            lookback: Number of recent trades to consider
            
        Returns:
            Win rate as decimal
        """
        results = self.risk_data.get('trade_results', [])
        if len(results) < 5:  # Need minimum trades
            return self.base_win_rate
        
        recent = results[-lookback:]
        wins = sum(1 for r in recent if r.get('is_win', False))
        return wins / len(recent)
    
    def get_risk_summary(self, current_capital: float) -> Dict:
        """
        Get a summary of current risk status.
        
        Args:
            current_capital: Current portfolio capital
            
        Returns:
            Dict with risk metrics
        """
        peak = self.risk_data.get('peak_capital', self.base_capital)
        drawdown = (current_capital - peak) / peak if peak > 0 else 0
        
        today = datetime.now().strftime('%Y-%m-%d')
        daily_pnl = self.risk_data.get('daily_pnl', {}).get(today, 0)
        weekly_pnl = self._get_weekly_pnl()
        
        rolling_wr = self.get_rolling_win_rate()
        total_trades = len(self.risk_data.get('trade_results', []))
        
        controls = self.check_drawdown_controls(current_capital)
        
        return {
            'current_capital': current_capital,
            'peak_capital': peak,
            'drawdown_pct': drawdown,
            'daily_pnl': daily_pnl,
            'weekly_pnl': weekly_pnl,
            'consecutive_losses': self.risk_data.get('consecutive_losses', 0),
            'rolling_win_rate': rolling_wr,
            'total_trades': total_trades,
            'trading_status': 'HALTED' if controls['halt_trading'] else 'ACTIVE',
            'size_reduction': controls['size_reduction'],
            'status_reason': controls['reason']
        }
    
    def print_risk_summary(self, current_capital: float):
        """Print formatted risk summary."""
        summary = self.get_risk_summary(current_capital)
        
        print("\n" + "=" * 60)
        print("RISK MANAGEMENT STATUS")
        print("=" * 60)
        print(f"  Trading Status: {summary['trading_status']}")
        print(f"  Status Reason: {summary['status_reason']}")
        print("-" * 60)
        print(f"  Current Capital: ${summary['current_capital']:,.2f}")
        print(f"  Peak Capital: ${summary['peak_capital']:,.2f}")
        print(f"  Drawdown: {summary['drawdown_pct']:.2%}")
        print("-" * 60)
        print(f"  Daily P&L: ${summary['daily_pnl']:,.2f}")
        print(f"  Weekly P&L: ${summary['weekly_pnl']:,.2f}")
        print(f"  Consecutive Losses: {summary['consecutive_losses']}")
        print("-" * 60)
        print(f"  Rolling Win Rate (20): {summary['rolling_win_rate']:.1%}")
        print(f"  Total Trades: {summary['total_trades']}")
        print(f"  Size Reduction: {summary['size_reduction']:.0%}")
        print("=" * 60)


class MultiStrategyRiskManager:
    """
    Manages multiple risk strategies (0.5x Kelly, 1.0x Kelly) in parallel.
    """
    
    def __init__(self, base_capital: float = 100000, data_dir: str = None):
        """
        Initialize multiple risk managers for different Kelly fractions.
        
        Args:
            base_capital: Starting capital for each strategy
            data_dir: Directory to store risk data
        """
        self.base_capital = base_capital
        self.data_dir = data_dir
        
        # Create risk managers for each strategy
        self.strategies = {
            'fixed': RiskManager(
                base_capital=base_capital,
                kelly_fraction=0,  # Will use fixed 2%
                data_dir=data_dir
            ),
            'kelly_0.5x': RiskManager(
                base_capital=base_capital,
                kelly_fraction=0.5,
                data_dir=data_dir
            ),
            'kelly_1.0x': RiskManager(
                base_capital=base_capital,
                kelly_fraction=1.0,
                data_dir=data_dir
            )
        }
        
        # Override file paths to keep separate
        for name, rm in self.strategies.items():
            rm.risk_data_file = os.path.join(
                data_dir or os.path.dirname(os.path.abspath(__file__)),
                'positions',
                f'risk_data_{name}.json'
            )
            rm.risk_data = rm._load_risk_data()
    
    def get_all_position_sizes(self, 
                                current_capitals: Dict[str, float],
                                ml_probability: float) -> Dict[str, Dict]:
        """
        Get position sizes for all strategies.
        
        Args:
            current_capitals: Dict of strategy_name -> current_capital
            ml_probability: ML model's probability
            
        Returns:
            Dict of strategy_name -> position_size_info
        """
        results = {}
        
        for name, rm in self.strategies.items():
            capital = current_capitals.get(name, self.base_capital)
            use_kelly = name != 'fixed'
            
            results[name] = rm.get_position_size(
                current_capital=capital,
                ml_probability=ml_probability,
                use_kelly=use_kelly
            )
            results[name]['strategy'] = name
        
        return results
    
    def record_trade_result_all(self, pnls: Dict[str, float], is_win: bool, date: str = None):
        """
        Record trade result for all strategies.
        
        Args:
            pnls: Dict of strategy_name -> pnl
            is_win: Whether the trade was profitable
            date: Date of the trade
        """
        for name, rm in self.strategies.items():
            pnl = pnls.get(name, 0)
            rm.record_trade_result(pnl, is_win, date)
    
    def print_all_summaries(self, current_capitals: Dict[str, float]):
        """Print risk summaries for all strategies."""
        print("\n" + "=" * 70)
        print("MULTI-STRATEGY RISK COMPARISON")
        print("=" * 70)
        
        headers = ['Metric', 'Fixed 2%', 'Kelly 0.5x', 'Kelly 1.0x']
        print(f"  {headers[0]:<20} {headers[1]:>15} {headers[2]:>15} {headers[3]:>15}")
        print("-" * 70)
        
        summaries = {}
        for name, rm in self.strategies.items():
            capital = current_capitals.get(name, self.base_capital)
            summaries[name] = rm.get_risk_summary(capital)
        
        metrics = [
            ('Capital', 'current_capital', '${:,.0f}'),
            ('Drawdown', 'drawdown_pct', '{:.1%}'),
            ('Daily P&L', 'daily_pnl', '${:,.0f}'),
            ('Weekly P&L', 'weekly_pnl', '${:,.0f}'),
            ('Win Rate', 'rolling_win_rate', '{:.1%}'),
            ('Consec. Losses', 'consecutive_losses', '{}'),
            ('Status', 'trading_status', '{}'),
        ]
        
        for label, key, fmt in metrics:
            vals = [
                fmt.format(summaries['fixed'][key]),
                fmt.format(summaries['kelly_0.5x'][key]),
                fmt.format(summaries['kelly_1.0x'][key])
            ]
            print(f"  {label:<20} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")
        
        print("=" * 70)


if __name__ == "__main__":
    # Test the risk manager
    rm = RiskManager(kelly_fraction=0.5)
    
    print("Kelly Criterion Test:")
    print(f"  Base win rate: {rm.base_win_rate:.1%}")
    print(f"  Kelly fraction: {rm.kelly_fraction}")
    print(f"  Kelly size: {rm.calculate_kelly_size():.1%}")
    
    print("\nConfidence Multipliers:")
    for prob in [0.60, 0.65, 0.75, 0.85, 0.90]:
        mult = rm.calculate_confidence_multiplier(prob)
        print(f"  {prob:.0%} confidence → {mult:.2f}x multiplier")
    
    print("\nPosition Size Calculation:")
    size_info = rm.get_position_size(
        current_capital=100000,
        ml_probability=0.75
    )
    print(f"  Capital: $100,000")
    print(f"  ML Probability: 75%")
    print(f"  Kelly %: {size_info['kelly_pct']:.1%}")
    print(f"  Confidence Mult: {size_info['confidence_multiplier']:.2f}x")
    print(f"  Final Size: ${size_info['position_size']:,.0f} ({size_info['position_pct']:.1%})")
    
    rm.print_risk_summary(100000)
