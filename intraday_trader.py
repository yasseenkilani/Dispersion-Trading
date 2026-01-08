"""
Intraday Paper Trader
=====================

Tracks positions opened via --log-data mode separately from the main portfolios.
Used to analyze whether entry timing affects profitability.

This is for RESEARCH purposes only - to compare:
- Morning entries vs. afternoon entries
- Multiple entries per day vs. single entry

Positions are held for 5 trading days, same as the main strategy.
"""

import os
import json
import logging
from datetime import datetime, timedelta

# Configuration
INTRADAY_POSITIONS_FILE = "intraday_data/intraday_positions.json"
INTRADAY_TRADES_FILE = "intraday_data/intraday_trades.csv"
HOLDING_PERIOD_DAYS = 5
POSITION_SIZE = 2000.0  # Same as main strategy
VEGA_MULTIPLIER = 0.15  # Same as main strategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class IntradayPaperTrader:
    """
    Paper trader for intraday signal tracking.
    
    Separate from main portfolios to avoid confusion.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.positions = []
        self.capital = 100000.0
        self._load_positions()
    
    def _load_positions(self):
        """Load existing positions from file."""
        os.makedirs("intraday_data", exist_ok=True)
        
        if os.path.exists(INTRADAY_POSITIONS_FILE):
            try:
                with open(INTRADAY_POSITIONS_FILE, 'r') as f:
                    data = json.load(f)
                    self.positions = data.get('positions', [])
                    self.capital = data.get('capital', 100000.0)
                    self.logger.info(f"[INTRADAY] Loaded {len(self.positions)} positions")
            except Exception as e:
                self.logger.error(f"[INTRADAY] Error loading positions: {e}")
                self.positions = []
    
    def _save_positions(self):
        """Save positions to file."""
        os.makedirs("intraday_data", exist_ok=True)
        
        with open(INTRADAY_POSITIONS_FILE, 'w') as f:
            json.dump({
                'positions': self.positions,
                'capital': self.capital,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def _count_trading_days(self, start_date, end_date):
        """Count trading days between two dates (excludes weekends)."""
        trading_days = 0
        current = start_date
        
        while current < end_date:
            if current.weekday() < 5:  # Monday to Friday
                trading_days += 1
            current += timedelta(days=1)
        
        return trading_days
    
    def _save_trade(self, trade):
        """Save closed trade to CSV history."""
        import pandas as pd
        
        df_row = pd.DataFrame([trade])
        
        if os.path.exists(INTRADAY_TRADES_FILE):
            history = pd.read_csv(INTRADAY_TRADES_FILE)
            history = pd.concat([history, df_row], ignore_index=True)
        else:
            history = df_row
        
        history.to_csv(INTRADAY_TRADES_FILE, index=False)
    
    def open_position(self, signal, ml_prediction):
        """
        Open a new intraday position if signal triggers.
        
        Returns the position if opened, None otherwise.
        """
        ml_signal = ml_prediction.get('ml_signal', 'NO_TRADE')
        
        # Only open if ML signals a trade
        if ml_signal not in ['SHORT_DISPERSION', 'LONG_DISPERSION']:
            return None
        
        # Create position record with timestamp
        timestamp = datetime.now()
        position = {
            'id': len(self.positions) + 1,
            'entry_timestamp': timestamp.isoformat(),
            'entry_date': timestamp.strftime('%Y-%m-%d'),
            'entry_time': timestamp.strftime('%H:%M:%S'),
            'exit_timestamp': None,
            'signal_type': ml_signal,
            'entry_corr': signal.get('impl_corr'),
            'entry_z_score': signal.get('z_score'),
            'entry_ml_prob': ml_prediction.get('short_probability'),
            'exit_corr': None,
            'position_size': POSITION_SIZE,
            'status': 'OPEN',
            'pnl': None,
            'holding_period': HOLDING_PERIOD_DAYS
        }
        
        self.positions.append(position)
        self._save_positions()
        
        return position
    
    def check_and_close_positions(self, current_corr=None):
        """
        Check for positions that have reached 5 trading days and close them.
        
        Returns list of closed positions.
        """
        closed = []
        
        for position in self.positions:
            if position['status'] != 'OPEN':
                continue
            
            entry_timestamp = datetime.fromisoformat(position['entry_timestamp'])
            trading_days_held = self._count_trading_days(entry_timestamp, datetime.now())
            
            if trading_days_held >= position['holding_period']:
                # Close the position
                position['exit_timestamp'] = datetime.now().isoformat()
                position['exit_corr'] = current_corr
                position['status'] = 'CLOSED'
                
                # Calculate P&L
                if position['entry_corr'] is not None and current_corr is not None:
                    corr_change = current_corr - position['entry_corr']
                    
                    if position['signal_type'] == 'SHORT_DISPERSION':
                        pnl = -corr_change * position['position_size'] * VEGA_MULTIPLIER
                    else:  # LONG_DISPERSION
                        pnl = corr_change * position['position_size'] * VEGA_MULTIPLIER
                    
                    position['pnl'] = pnl
                    self.capital += pnl
                else:
                    position['pnl'] = 0
                
                # Save to trade history
                self._save_trade({
                    'id': position['id'],
                    'entry_timestamp': position['entry_timestamp'],
                    'exit_timestamp': position['exit_timestamp'],
                    'signal_type': position['signal_type'],
                    'entry_corr': position['entry_corr'],
                    'exit_corr': position['exit_corr'],
                    'entry_ml_prob': position['entry_ml_prob'],
                    'position_size': position['position_size'],
                    'pnl': position['pnl']
                })
                
                closed.append(position)
        
        if closed:
            self._save_positions()
        
        return closed
    
    def get_stats(self):
        """Get portfolio statistics."""
        open_positions = [p for p in self.positions if p['status'] == 'OPEN']
        closed_positions = [p for p in self.positions if p['status'] == 'CLOSED']
        
        total_pnl = sum(p.get('pnl', 0) or 0 for p in closed_positions)
        wins = sum(1 for p in closed_positions if (p.get('pnl') or 0) > 0)
        win_rate = wins / len(closed_positions) * 100 if closed_positions else 0
        
        return {
            'capital': self.capital,
            'open_positions': len(open_positions),
            'closed_positions': len(closed_positions),
            'total_pnl': total_pnl,
            'win_rate': win_rate
        }
    
    def print_summary(self):
        """Print intraday trading summary."""
        stats = self.get_stats()
        open_positions = [p for p in self.positions if p['status'] == 'OPEN']
        closed_positions = [p for p in self.positions if p['status'] == 'CLOSED']
        
        print("\n" + "─" * 60)
        print("📊 INTRADAY SIGNAL TRACKING")
        print("─" * 60)
        
        # Open positions
        if open_positions:
            print("\nOpen Positions:")
            for pos in open_positions:
                entry_ts = datetime.fromisoformat(pos['entry_timestamp'])
                trading_days = self._count_trading_days(entry_ts, datetime.now())
                remaining = pos['holding_period'] - trading_days
                print(f"  #{pos['id']}: {pos['entry_date']} {pos['entry_time'][:5]} | "
                      f"{pos['signal_type'][:5]} | Entry: {pos['entry_corr']:.4f} | "
                      f"Days: {trading_days}/5 | Closes in: {remaining}")
        else:
            print("\nOpen Positions: None")
        
        # Recent closed positions (last 5)
        if closed_positions:
            print("\nRecent Closed Positions:")
            for pos in closed_positions[-5:]:
                pnl = pos.get('pnl', 0) or 0
                pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                status = "✅" if pnl > 0 else "❌"
                print(f"  #{pos['id']}: {pos['entry_date']} {pos['entry_time'][:5]} | "
                      f"{pos['signal_type'][:5]} | Entry: {pos['entry_corr']:.4f} | "
                      f"Exit: {pos['exit_corr']:.4f} | P&L: {pnl_str} {status}")
        
        # Summary
        print(f"\n{'─' * 60}")
        print(f"Summary: {stats['open_positions']} open | {stats['closed_positions']} closed | "
              f"Win Rate: {stats['win_rate']:.1f}% | Total P&L: ${stats['total_pnl']:.2f}")
        print("─" * 60)


def run_intraday_tracking(signal, ml_prediction, current_corr=None):
    """
    Main function to run intraday tracking.
    
    Called from run_daily.py when --log-data is used.
    """
    trader = IntradayPaperTrader()
    
    # First, check and close any expired positions
    if current_corr is None:
        current_corr = signal.get('impl_corr')
    
    closed = trader.check_and_close_positions(current_corr)
    if closed:
        print(f"\n✅ Closed {len(closed)} intraday position(s)")
        for pos in closed:
            pnl = pos.get('pnl', 0) or 0
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            print(f"   #{pos['id']}: {pos['signal_type']} | P&L: {pnl_str}")
    
    # Then, try to open a new position if signal triggers
    position = trader.open_position(signal, ml_prediction)
    if position:
        print(f"\n📈 Opened intraday position #{position['id']}")
        print(f"   Signal: {position['signal_type']}")
        print(f"   Entry Correlation: {position['entry_corr']:.4f}")
        print(f"   ML Probability: {position['entry_ml_prob']:.1%}")
    
    # Print summary
    trader.print_summary()
    
    return trader
