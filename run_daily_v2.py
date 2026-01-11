"""
Dispersion Trading - Daily Runner v2
=====================================

Enhanced version with multi-portfolio risk management:
- Fixed 2% (Z-score baseline)
- Kelly 0.5x + Confidence Scaling (ML)
- Kelly 1.0x + Confidence Scaling (ML)

Includes:
- Kelly criterion position sizing
- Confidence-based allocation scaling
- Drawdown controls and circuit breakers
- VEGA-WEIGHTED position sizing (optional)

Usage:
    python run_daily_v2.py              # Full run with IBKR
    python run_daily_v2.py --offline    # Test mode without IBKR
    python run_daily_v2.py --signal-only # Generate signal only, no trading
    python run_daily_v2.py --log-data   # Intraday data logging with vega-weighted sizing
    python run_daily_v2.py --no-vega    # Disable vega-weighted sizing
"""

import argparse
import sys
import os
from datetime import datetime
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_generator import DispersionSignalGenerator, run_offline_test
from multi_portfolio_tracker import MultiPortfolioTrader, run_multi_portfolio_session
from intraday_trader import run_intraday_tracking
from correlation_calculator import ImpliedCorrelationCalculator
from ibkr_connector import INDEX_SYMBOL

# Try to import ML predictor
try:
    from ml_predictor import MLPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Try to import vega-weighted sizer
try:
    from vega_weighted_sizer import VegaWeightedKellySizer, MultiStrategyVegaSizer, VEGA_SIZER_AVAILABLE
    VEGA_AVAILABLE = VEGA_SIZER_AVAILABLE
except ImportError:
    VEGA_AVAILABLE = False

# Signal history file
SIGNAL_HISTORY_FILE = "signals/signal_history.csv"

# Intraday data folder
INTRADAY_DATA_FOLDER = "intraday_data"


def save_intraday_data(signal, ml_prediction, iv_data, vega_info=None):
    """Save intraday data snapshot for research purposes."""
    import json
    
    os.makedirs(INTRADAY_DATA_FOLDER, exist_ok=True)
    
    timestamp = datetime.now()
    filename = f"{timestamp.strftime('%Y-%m-%d_%H-%M')}.json"
    filepath = os.path.join(INTRADAY_DATA_FOLDER, filename)
    
    data = {
        'timestamp': timestamp.isoformat(),
        'date': timestamp.strftime('%Y-%m-%d'),
        'time': timestamp.strftime('%H:%M:%S'),
        'market_data': {
            'impl_corr': signal.get('impl_corr'),
            'z_score': signal.get('z_score'),
            'index_iv': signal.get('index_iv'),
            'num_components': signal.get('num_components'),
            'vix_level': iv_data.get('vix_level'),
            'vxn_level': iv_data.get('vxn_level')
        },
        'signals': {
            'z_score_signal': signal.get('signal'),
            'ml_signal': ml_prediction.get('ml_signal'),
            'short_probability': ml_prediction.get('short_probability'),
            'strategy_used': ml_prediction.get('strategy_used')
        },
        'vega_info': vega_info,
        'percentile': signal.get('percentile'),
        'historical_mean': signal.get('historical_mean'),
        'historical_std': signal.get('historical_std')
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    return filepath


def get_intraday_summary():
    """Get summary of intraday data collected today."""
    import json
    
    today = datetime.now().strftime('%Y-%m-%d')
    snapshots = []
    
    if os.path.exists(INTRADAY_DATA_FOLDER):
        for filename in sorted(os.listdir(INTRADAY_DATA_FOLDER)):
            if filename.startswith(today) and filename.endswith('.json'):
                filepath = os.path.join(INTRADAY_DATA_FOLDER, filename)
                with open(filepath, 'r') as f:
                    snapshots.append(json.load(f))
    
    return snapshots


def load_historical_data():
    """Load historical correlation data for ML features."""
    historical_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'historical_data', 'correlation_data.csv'
    )
    
    if os.path.exists(historical_file):
        df = pd.read_csv(historical_file, parse_dates=['date'])
        df = df.set_index('date')
        return df
    return None


def save_signal_to_history(signal, ml_prediction):
    """Save both Z-score and ML signals to history file."""
    os.makedirs("signals", exist_ok=True)
    
    # Prepare row with both signals
    row = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().isoformat(),
        'impl_corr': signal.get('impl_corr'),
        'z_score': signal.get('z_score'),
        'z_signal': signal.get('signal'),
        'short_probability': ml_prediction.get('short_probability'),
        'ml_signal': ml_prediction.get('ml_signal'),
        'strategy_used': ml_prediction.get('strategy_used', 'N/A'),
        'agreement': 'AGREE' if signal.get('signal') == ml_prediction.get('ml_signal') else 'DIVERGE',
        'index_iv': signal.get('index_iv'),
        'num_components': signal.get('num_components')
    }
    
    df_row = pd.DataFrame([row])
    
    if os.path.exists(SIGNAL_HISTORY_FILE):
        history = pd.read_csv(SIGNAL_HISTORY_FILE)
        # Check if we already have an entry for today
        today = datetime.now().strftime('%Y-%m-%d')
        if today in history['date'].values:
            # Update today's entry
            history = history[history['date'] != today]
        history = pd.concat([history, df_row], ignore_index=True)
    else:
        history = df_row
    
    history.to_csv(SIGNAL_HISTORY_FILE, index=False)
    return len(history)


def print_signal_comparison(signal, ml_prediction):
    """Print a side-by-side comparison of Z-score and ML hybrid signals."""
    print("\n" + "=" * 70)
    print("SIGNAL COMPARISON: Z-SCORE vs HYBRID ML")
    print("=" * 70)
    
    z_score = signal.get('z_score', 0)
    impl_corr = signal.get('impl_corr', 0)
    z_signal = signal.get('signal', 'NO_TRADE')
    
    short_prob = ml_prediction.get('short_probability')
    ml_signal = ml_prediction.get('ml_signal', 'N/A')
    strategy_used = ml_prediction.get('strategy_used', 'N/A')
    
    print(f"\n┌{'─' * 68}┐")
    print(f"│{'CURRENT MARKET CONDITIONS':^68}│")
    print(f"├{'─' * 34}┬{'─' * 33}┤")
    print(f"│ {'Implied Correlation:':<32} │ {impl_corr:>31.4f} │")
    print(f"│ {'Z-Score:':<32} │ {z_score:>31.4f} │")
    print(f"├{'─' * 34}┴{'─' * 33}┤")
    print(f"│{'':^68}│")
    print(f"├{'─' * 34}┬{'─' * 33}┤")
    print(f"│{'Z-SCORE ONLY':^34}│{'HYBRID ML STRATEGY':^33}│")
    print(f"├{'─' * 34}┼{'─' * 33}┤")
    print(f"│ {'Rules:':<32} │ {'Rules:':<31} │")
    print(f"│ {'  Z > 1.5 → SHORT':<32} │ {'  SHORT: ML prob ≥ 60%':<31} │")
    print(f"│ {'  Z < -1.5 → LONG':<32} │ {'  LONG: Z-score < -1.5':<31} │")
    print(f"│ {'  else → NO_TRADE':<32} │ {'  (ML not used for LONG)':<31} │")
    print(f"├{'─' * 34}┼{'─' * 33}┤")
    
    # Format Z-score signal
    if z_signal == 'SHORT_DISPERSION':
        z_display = f"🔴 SHORT"
    elif z_signal == 'LONG_DISPERSION':
        z_display = f"🟢 LONG"
    else:
        z_display = f"⚪ NO_TRADE"
    
    # Format ML hybrid signal
    if short_prob is not None:
        prob_str = f"{short_prob:.1%}"
        if ml_signal == 'SHORT_DISPERSION':
            ml_display = f"🔴 SHORT ({prob_str})"
            strategy_display = f"via ML"
        elif ml_signal == 'LONG_DISPERSION':
            ml_display = f"🟢 LONG (Z: {z_score:.2f})"
            strategy_display = f"via Z-score"
        else:
            ml_display = f"⚪ NO_TRADE ({prob_str})"
            strategy_display = f"--"
    else:
        ml_display = "❌ NOT AVAILABLE"
        strategy_display = "--"
    
    print(f"│ {'Signal:':<32} │ {'Signal:':<31} │")
    print(f"│ {z_display:<32} │ {ml_display:<31} │")
    print(f"│ {'(Z-score threshold only)':<32} │ {f'Strategy: {strategy_display}':<31} │")
    print(f"├{'─' * 34}┴{'─' * 33}┤")
    
    # Agreement/Divergence and explanation
    if short_prob is not None:
        if z_signal == ml_signal:
            if ml_signal == 'SHORT_DISPERSION':
                verdict = "✅ AGREEMENT: Both recommend SHORT"
            elif ml_signal == 'LONG_DISPERSION':
                verdict = "✅ AGREEMENT: Both recommend LONG"
            else:
                verdict = "✅ AGREEMENT: Both recommend NO TRADE"
        else:
            if ml_signal == 'SHORT_DISPERSION' and z_signal == 'NO_TRADE':
                verdict = "⚠️  ML EARLY ENTRY: ML sees SHORT opportunity"
            elif ml_signal == 'NO_TRADE' and z_signal == 'SHORT_DISPERSION':
                verdict = "⚠️  ML FILTER: Z-score triggers but ML says wait"
            else:
                verdict = f"⚠️  DIVERGENCE: Z={z_signal}, ML={ml_signal}"
    else:
        verdict = "ℹ️  ML model not available for comparison"
    
    print(f"│{verdict:^68}│")
    print(f"└{'─' * 68}┘")
    
    # Print ML explanation
    if short_prob is not None:
        print(f"\n📊 ML Analysis:")
        print(f"   SHORT probability: {short_prob:.1%}")
        print(f"   ML threshold: ≥60% for SHORT signal")
        print(f"   LONG uses Z-score only (ML doesn't improve LONG predictions)")


def print_risk_management_info(use_vega: bool = True):
    """Print risk management configuration."""
    print("\n" + "-" * 70)
    print("RISK MANAGEMENT CONFIGURATION")
    print("-" * 70)
    
    print("\n  Position Sizing:")
    print("    Fixed 2%:    Base 2% per trade (Z-score signal)")
    print("    Kelly 0.5x:  Half Kelly + confidence scaling (ML signal)")
    print("    Kelly 1.0x:  Full Kelly + confidence scaling (ML signal)")
    
    print("\n  Confidence Multipliers (based on ML probability):")
    print("    ≥85%: 1.50x | ≥75%: 1.25x | ≥65%: 1.00x | ≥60%: 0.75x | <60%: 0.50x")
    
    print("\n  Drawdown Controls:")
    print("    Daily:  -2% → halt new trades")
    print("    Weekly: -5% → reduce size 50%")
    print("    Monthly: -10% → halt trading")
    print("    Consecutive losses: 5 → reduce size 50%")
    
    print("\n  Vega-Weighted Sizing:")
    if use_vega and VEGA_AVAILABLE:
        print("    Status: ✅ ENABLED")
        print("    Index/Component split: 50%/50%")
        print("    Component weighting: Inverse-vega (higher vega = smaller position)")
        print("    Goal: Vega-neutral exposure (no directional vol bet)")
    else:
        print("    Status: ❌ DISABLED")
        print("    Using simple Kelly-based sizing without vega weighting")


def main():
    """Main entry point for daily dispersion trading with risk management."""
    parser = argparse.ArgumentParser(description='Dispersion Trading Daily Runner v2 (Multi-Portfolio)')
    parser.add_argument('--offline', action='store_true', 
                        help='Run in offline test mode (no IBKR connection)')
    parser.add_argument('--signal-only', action='store_true',
                        help='Generate signal only, do not execute trades')
    parser.add_argument('--port', type=int, default=7497,
                        help='IBKR port (7497=paper, 7496=live)')
    parser.add_argument('--check-positions', action='store_true',
                        help='Only check and close expired positions')
    parser.add_argument('--no-ml', action='store_true',
                        help='Disable ML predictions')
    parser.add_argument('--log-data', action='store_true',
                        help='Log intraday data only (no trades, can run multiple times)')
    parser.add_argument('--show-risk', action='store_true',
                        help='Show risk management configuration')
    parser.add_argument('--no-vega', action='store_true',
                        help='Disable vega-weighted position sizing')
    parser.add_argument('--live-greeks', action='store_true',
                        help='Use live IBKR data for Greeks (requires IBKR connection)')
    
    args = parser.parse_args()
    
    # Determine if vega weighting is enabled
    use_vega = not args.no_vega and VEGA_AVAILABLE
    use_ibkr_greeks = args.live_greeks and not args.offline
    
    print("\n" + "=" * 90)
    print("DISPERSION TRADING SYSTEM v2 - MULTI-PORTFOLIO RISK MANAGEMENT")
    print("=" * 90)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Index: {INDEX_SYMBOL} (QQQ-based)")
    print(f"Mode: {'OFFLINE TEST' if args.offline else 'LIVE (IBKR)'}")
    print(f"Strategy: HYBRID (ML for SHORT, Z-score for LONG)")
    print(f"ML Status: {'DISABLED' if args.no_ml else ('ENABLED' if ML_AVAILABLE else 'NOT AVAILABLE')}")
    print(f"Vega-Weighted: {'✅ ENABLED' if use_vega else '❌ DISABLED'}")
    if use_vega:
        print(f"Greeks Source: {'LIVE (IBKR)' if use_ibkr_greeks else 'SIMULATED'}")
    print("=" * 90)
    
    # Show risk management info
    if args.show_risk:
        print_risk_management_info(use_vega)
        return
    
    # Check positions only
    if args.check_positions:
        print("\nChecking positions only...")
        trader = MultiPortfolioTrader()
        trader.print_full_summary()
        return
    
    # Log data mode header
    if args.log_data:
        print("\n📊 INTRADAY DATA LOGGING MODE")
        print("   Collecting data for research (no trades will be executed)")
        print("   This data is NOT used for ML training to avoid overfitting")
        if use_vega:
            print("   📈 Vega-weighted sizing ENABLED for all 3 portfolios")
    
    # Generate signal
    if args.offline:
        signal = run_offline_test()
        iv_data = {
            'index_iv': 18.5,
            'vix_level': 18.0,
            'vxn_level': 20.0
        }
    else:
        generator = DispersionSignalGenerator()
        signal = generator.run()
        iv_data = {
            'index_iv': signal.get('index_iv', 20) if signal else 20,
            'vix_level': 18.0,
            'vxn_level': 20.0
        }
    
    if signal is None:
        print("\nFailed to generate signal")
        return
    
    # Generate ML prediction
    ml_prediction = {
        'ml_available': False, 
        'short_probability': None, 
        'ml_signal': 'NO_TRADE',
        'strategy_used': 'N/A'
    }
    
    if ML_AVAILABLE and not args.no_ml:
        try:
            predictor = MLPredictor()
            
            if predictor.loaded:
                # Load historical data for feature calculation
                historical_data = load_historical_data()
                
                # Get ML prediction
                ml_prediction = predictor.predict(
                    iv_data=iv_data,
                    z_score=signal.get('z_score', 0),
                    impl_corr=signal.get('impl_corr', 0),
                    historical_data=historical_data
                )
        except Exception as e:
            print(f"\n⚠️  ML prediction error: {e}")
    
    # Print signal comparison
    print_signal_comparison(signal, ml_prediction)
    
    # Calculate vega info if enabled
    vega_info = None
    if use_vega and VEGA_AVAILABLE:
        try:
            from vega_weighted_sizer import MultiStrategyVegaSizer
            vega_sizer = MultiStrategyVegaSizer(base_capital=100000)
            
            ml_signal = ml_prediction.get('ml_signal', 'NO_TRADE')
            z_signal = signal.get('signal', 'NO_TRADE')
            
            if ml_signal in ['SHORT_DISPERSION', 'LONG_DISPERSION']:
                signal_type = ml_signal
            elif z_signal in ['SHORT_DISPERSION', 'LONG_DISPERSION']:
                signal_type = z_signal
            else:
                signal_type = 'SHORT_DISPERSION'  # Default for display
            
            vega_positions = vega_sizer.calculate_all_positions(
                signal_type=signal_type,
                current_capitals={'fixed': 100000, 'kelly_0.5x': 100000, 'kelly_1.0x': 100000},
                ml_probability=ml_prediction.get('short_probability', 0.5),
                use_ibkr=use_ibkr_greeks
            )
            
            # Print vega comparison
            vega_sizer.print_comparison(vega_positions)
            
            # Store vega info for logging
            vega_info = {
                strategy: {
                    'total_position': pos.get('total_position', 0),
                    'is_vega_neutral': pos.get('summary', {}).get('is_vega_neutral', False),
                    'vega_ratio': pos.get('summary', {}).get('vega_ratio', 1.0)
                }
                for strategy, pos in vega_positions.items()
            }
        except Exception as e:
            print(f"\n⚠️  Vega calculation error: {e}")
    
    # Handle log-data mode
    if args.log_data:
        # Save intraday snapshot
        filepath = save_intraday_data(signal, ml_prediction, iv_data, vega_info)
        print(f"\n✅ Intraday data saved: {filepath}")
        
        # Show today's snapshots
        snapshots = get_intraday_summary()
        print(f"\n📊 Today's data snapshots: {len(snapshots)}")
        if snapshots:
            print("   Time       | Impl Corr | Z-Score | ML Prob  | ML Signal")
            print("   " + "-" * 60)
            for snap in snapshots:
                time = snap.get('time', 'N/A')[:5]
                corr = snap['market_data'].get('impl_corr', 0)
                z = snap['market_data'].get('z_score', 0)
                prob = snap['signals'].get('short_probability')
                ml_sig = snap['signals'].get('ml_signal', 'N/A')
                prob_str = f"{prob:.1%}" if prob else "N/A"
                print(f"   {time}     | {corr:.4f}  | {z:>7.4f} | {prob_str:>8} | {ml_sig}")
        
        # Run intraday paper trading with vega-weighted sizing
        run_intraday_tracking(
            signal, 
            ml_prediction,
            use_vega_weighting=use_vega,
            use_ibkr=use_ibkr_greeks
        )
        
        print("\nℹ️  Main portfolios NOT affected (intraday tracking is separate)")
        print("   Run without --log-data to execute main strategy trades")
    else:
        # Save both signals to history
        num_signals = save_signal_to_history(signal, ml_prediction)
        print(f"\n📊 Signal history updated: {num_signals} total signals logged")
        
        # Execute trades for all 3 strategies if not signal-only mode
        if not args.signal_only:
            run_multi_portfolio_session(signal, ml_prediction)
        else:
            print("\nSignal-only mode - no trades executed")
            # Still show portfolio status
            trader = MultiPortfolioTrader()
            trader.print_full_summary()
    
    print("\n" + "=" * 90)
    print("DAILY RUN COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
