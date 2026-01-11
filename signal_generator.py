"""
Dispersion Trading Daily Signal Generator
==========================================

Main script that:
1. Connects to IBKR
2. Fetches real-time IV data (QQQ + components)
3. Calculates implied correlation
4. Generates trading signal
5. Logs results and sends alerts

UPDATED: Now uses QQQ instead of NDX for better liquidity.

Run daily after market open to generate trading signals.
"""

import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ibkr_connector import IBKRConnector, NDX_COMPONENTS, INDEX_SYMBOL
from correlation_calculator import ImpliedCorrelationCalculator

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
OUTPUT_DIR = "signals"
LOG_DIR = "logs"
HISTORICAL_DATA_DIR = "historical_data"

# Alert settings
ENABLE_EMAIL_ALERTS = False  # Set to True to enable email alerts
EMAIL_RECIPIENT = "your_email@example.com"

# Safety killswitch
MIN_COMPONENTS_REQUIRED = 45  # Minimum components needed for valid signal

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Set up logging configuration."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    log_file = os.path.join(LOG_DIR, f"signal_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class DispersionSignalGenerator:
    """
    Main class for generating dispersion trading signals.
    
    Uses QQQ as the index proxy instead of NDX for:
    - Better liquidity (40x more volume)
    - Tighter spreads (97% lower)
    - Smaller contract sizes
    """
    
    def __init__(self):
        self.logger = setup_logging()
        self.ibkr = None
        self.calculator = None
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)
    
    def initialize(self):
        """Initialize IBKR connection and calculator."""
        self.logger.info("Initializing signal generator...")
        self.logger.info(f"Index Symbol: {INDEX_SYMBOL}")
        
        # Initialize IBKR connector
        self.ibkr = IBKRConnector()
        
        # Initialize correlation calculator
        self.calculator = ImpliedCorrelationCalculator(HISTORICAL_DATA_DIR)
        
        return True
    
    def connect_ibkr(self):
        """Connect to IBKR."""
        self.logger.info("Connecting to IBKR...")
        
        if self.ibkr.connect():
            self.logger.info("Successfully connected to IBKR")
            return True
        else:
            self.logger.error("Failed to connect to IBKR")
            return False
    
    def disconnect_ibkr(self):
        """Disconnect from IBKR."""
        if self.ibkr:
            self.ibkr.disconnect()
            self.logger.info("Disconnected from IBKR")
    
    def fetch_iv_data(self):
        """Fetch IV data from IBKR."""
        self.logger.info("Fetching IV data from IBKR...")
        
        # Get QQQ IV (will fall back to VXN if not available)
        index_iv = self.ibkr.get_index_iv(INDEX_SYMBOL)
        if index_iv:
            self.logger.info(f"{INDEX_SYMBOL} IV: {index_iv:.2f}%")
        else:
            # Try VXN directly as fallback
            self.logger.warning(f"{INDEX_SYMBOL} IV not available, trying VXN...")
            index_iv = self.ibkr.get_vxn_level()
            if index_iv:
                self.logger.info(f"Using VXN as proxy: {index_iv:.2f}%")
            else:
                # Last resort: VIX
                self.logger.warning("VXN not available, trying VIX...")
                index_iv = self.ibkr.get_vix_level()
                if index_iv:
                    self.logger.info(f"Using VIX as proxy: {index_iv:.2f}%")
                else:
                    self.logger.error("Could not fetch any index IV data")
                    return None
        
        # Get component IVs
        component_ivs = self.ibkr.get_all_component_ivs()
        num_components = len(component_ivs)
        self.logger.info(f"Got IV for {num_components} components")
        
        # KILLSWITCH: Check minimum components requirement
        if num_components < MIN_COMPONENTS_REQUIRED:
            self.logger.error(f"\n{'='*60}")
            self.logger.error(f"🚨 KILLSWITCH ACTIVATED 🚨")
            self.logger.error(f"{'='*60}")
            self.logger.error(f"Only got IV for {num_components}/{len(NDX_COMPONENTS)} components")
            self.logger.error(f"Minimum required: {MIN_COMPONENTS_REQUIRED}")
            self.logger.error(f"")
            self.logger.error(f"⚠️  DATA QUALITY ISSUE - TRADE NOT TAKEN")
            self.logger.error(f"Please rerun the code and ensure IBKR connection is uninterrupted")
            self.logger.error(f"{'='*60}\n")
            return None
        
        # Get weights
        weights = {symbol: weight for symbol, weight in NDX_COMPONENTS}
        
        return {
            'index_iv': index_iv,
            'qqq_iv': index_iv,
            'ndx_iv': index_iv,  # Backward compatibility
            'component_ivs': component_ivs,
            'weights': weights,
            'index_symbol': INDEX_SYMBOL
        }
    
    def generate_signal(self, iv_data):
        """Generate trading signal from IV data."""
        self.logger.info("Generating trading signal...")
        
        signal = self.calculator.generate_signal(
            iv_data['index_iv'],
            iv_data['component_ivs'],
            iv_data['weights']
        )
        
        self.logger.info(f"Signal: {signal['signal']}")
        self.logger.info(f"Implied Correlation: {signal.get('impl_corr', 'N/A')}")
        self.logger.info(f"Z-Score: {signal.get('z_score', 'N/A')}")
        
        return signal
    
    def save_signal(self, signal, iv_data):
        """Save signal to file."""
        timestamp = datetime.now()
        date_str = timestamp.strftime('%Y%m%d')
        
        # Save signal summary
        signal_file = os.path.join(OUTPUT_DIR, f"signal_{date_str}.json")
        
        signal_data = {
            'timestamp': timestamp.isoformat(),
            'date': date_str,
            **signal,
            'index_symbol': iv_data.get('index_symbol', INDEX_SYMBOL),
            'index_iv': iv_data['index_iv'],
            'qqq_iv': iv_data.get('qqq_iv', iv_data['index_iv']),
            'num_components': len(iv_data['component_ivs'])
        }
        
        with open(signal_file, 'w') as f:
            json.dump(signal_data, f, indent=2, default=str)
        
        self.logger.info(f"Signal saved to {signal_file}")
        
        # Append to signal history
        history_file = os.path.join(OUTPUT_DIR, "signal_history.csv")
        
        history_row = pd.DataFrame([{
            'date': date_str,
            'timestamp': timestamp.isoformat(),
            'signal': signal['signal'],
            'impl_corr': signal.get('impl_corr'),
            'z_score': signal.get('z_score'),
            'index_symbol': iv_data.get('index_symbol', INDEX_SYMBOL),
            'index_iv': iv_data['index_iv'],
            'num_components': len(iv_data['component_ivs'])
        }])
        
        if os.path.exists(history_file):
            history = pd.read_csv(history_file)
            history = pd.concat([history, history_row], ignore_index=True)
        else:
            history = history_row
        
        history.to_csv(history_file, index=False)
        self.logger.info(f"Signal history updated: {len(history)} total signals")
        
        # Update historical correlation data
        if signal.get('impl_corr') is not None:
            self.calculator.update_historical_data(signal['impl_corr'])
        
        return signal_file
    
    def send_alert(self, signal):
        """Send alert for actionable signals."""
        if signal['signal'] in ['LONG_DISPERSION', 'SHORT_DISPERSION']:
            alert_msg = f"""
========================================
DISPERSION TRADING ALERT
========================================

Signal: {signal['signal']}
Time: {signal['timestamp']}
Index: {INDEX_SYMBOL}

Implied Correlation: {signal.get('impl_corr', 'N/A'):.4f}
Z-Score: {signal.get('z_score', 'N/A'):.4f}

Reason: {signal.get('reason', 'N/A')}

ACTION REQUIRED: Review and execute trade
Trade {INDEX_SYMBOL} options (not NDX)
========================================
"""
            self.logger.info(alert_msg)
            
            # Save alert to file
            alert_file = os.path.join(OUTPUT_DIR, f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(alert_file, 'w') as f:
                f.write(alert_msg)
            
            # TODO: Add email/SMS alert integration
            if ENABLE_EMAIL_ALERTS:
                self.logger.info(f"Email alert would be sent to {EMAIL_RECIPIENT}")
        else:
            self.logger.info("No actionable signal - no alert sent")
    
    def run(self):
        """Run the complete signal generation process."""
        print("\n" + "=" * 60)
        print("DISPERSION TRADING SIGNAL GENERATOR")
        print(f"Index: {INDEX_SYMBOL} (QQQ-based)")
        print("=" * 60)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        try:
            # Initialize
            self.initialize()
            
            # Connect to IBKR
            if not self.connect_ibkr():
                self.logger.error("Cannot proceed without IBKR connection")
                return None
            
            # Fetch IV data
            iv_data = self.fetch_iv_data()
            if iv_data is None:
                self.logger.error("Cannot proceed without IV data")
                self.disconnect_ibkr()
                return None
            
            # Generate signal
            signal = self.generate_signal(iv_data)
            
            # Save signal
            self.save_signal(signal, iv_data)
            
            # Send alert if actionable
            self.send_alert(signal)
            
            # Disconnect
            self.disconnect_ibkr()
            
            print("\n" + "=" * 60)
            print("SIGNAL GENERATION COMPLETE")
            print("=" * 60)
            print(f"\nIndex: {INDEX_SYMBOL}")
            print(f"Final Signal: {signal['signal']}")
            if signal.get('impl_corr'):
                print(f"Implied Correlation: {signal['impl_corr']:.4f}")
            if signal.get('z_score'):
                print(f"Z-Score: {signal['z_score']:.4f}")
            print("=" * 60)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error during signal generation: {e}")
            import traceback
            traceback.print_exc()
            self.disconnect_ibkr()
            return None


# =============================================================================
# OFFLINE MODE (for testing without IBKR)
# =============================================================================

def run_offline_test():
    """Run signal generator in offline mode with sample data."""
    print("\n" + "=" * 60)
    print("OFFLINE TEST MODE")
    print(f"Index: {INDEX_SYMBOL} (QQQ-based)")
    print("=" * 60)
    
    # Initialize calculator
    calculator = ImpliedCorrelationCalculator(HISTORICAL_DATA_DIR)
    
    # Sample IV data (simulating market conditions)
    # Using typical QQQ IV levels (slightly lower than NDX due to ETF structure)
    index_iv = 18.5  # QQQ IV
    
    component_ivs = {
        'NVDA': 45.2, 'AAPL': 22.1, 'GOOGL': 25.3, 'GOOG': 25.1,
        'MSFT': 21.5, 'AMZN': 28.7, 'META': 32.4, 'AVGO': 35.6,
        'TSLA': 55.2, 'NFLX': 38.9, 'ASML': 30.2, 'COST': 18.5,
        'AMD': 48.3, 'CSCO': 19.2, 'MU': 42.1, 'TMUS': 22.8,
        'AMAT': 35.4, 'PEP': 15.2, 'LRCX': 38.7, 'ISRG': 25.6,
        'QCOM': 32.1, 'INTU': 28.4, 'INTC': 35.8, 'BKNG': 30.2,
        'AMGN': 20.5, 'TXN': 25.3, 'KLAC': 36.2, 'PDD': 52.1,
        'GILD': 22.4, 'ADBE': 30.8,
    }
    
    weights = {symbol: weight for symbol, weight in NDX_COMPONENTS}
    
    # Generate signal
    signal = calculator.generate_signal(index_iv, component_ivs, weights)
    
    print("\n" + "=" * 60)
    print("OFFLINE TEST RESULT")
    print("=" * 60)
    print(f"Index: {INDEX_SYMBOL}")
    for key, value in signal.items():
        print(f"  {key}: {value}")
    
    return signal


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dispersion Trading Signal Generator (QQQ-based)')
    parser.add_argument('--offline', action='store_true', help='Run in offline test mode')
    parser.add_argument('--port', type=int, default=7497, help='IBKR port (7497=paper, 7496=live)')
    
    args = parser.parse_args()
    
    if args.offline:
        run_offline_test()
    else:
        generator = DispersionSignalGenerator()
        generator.run()


if __name__ == "__main__":
    main()
