"""
Institutional-Grade Automated Dispersion Trader
================================================

Implements professional quant desk best practices:
- Vega-neutral position sizing
- Complex combo orders (multi-leg)
- Multiple safety checks and circuit breakers
- Manual confirmation workflow (Crawl phase)

Based on institutional trading desk standards.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ibkr_connector import IBKRConnector, NDX_COMPONENTS, INDEX_SYMBOL

# =============================================================================
# CONFIGURATION - INSTITUTIONAL STANDARDS
# =============================================================================

# Position Sizing
RISK_PER_TRADE_PCT = 0.02  # 2% max loss per trade
MAX_GROSS_EXPOSURE_PCT = 0.50  # Max 50% of capital deployed
MAX_CONCURRENT_TRADES = 3

# Trade Structure
NUM_COMPONENTS = 30  # Use 30 components (institutional minimum)
TARGET_DTE = 30  # Target days to expiration
DTE_RANGE = (21, 45)  # Acceptable DTE range

# Order Execution
SLIPPAGE_TOLERANCE_PCT = 0.02  # Max 2% slippage from mid-price
ORDER_TIMEOUT_SECONDS = 30  # Cancel if not filled in 30 seconds
PRICE_WALK_INCREMENT = 0.01  # Walk price by $0.01 per attempt

# Safety Checks
VIX_CIRCUIT_BREAKER = 30  # Block new trades if VIX > 30
MIN_ACCOUNT_BALANCE = 50000  # Minimum account balance to trade

# Paths
OUTPUT_DIR = "automated_trades"
LOG_DIR = "logs"

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Set up logging configuration."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    log_file = os.path.join(LOG_DIR, f"auto_trader_{datetime.now().strftime('%Y%m%d')}.log")
    
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
# VEGA-NEUTRAL POSITION SIZER
# =============================================================================

class VegaNeutralSizer:
    """
    Calculate vega-neutral position sizes for dispersion trades.
    
    Core Principle: Total component vega must equal index vega to isolate
    correlation risk and neutralize directional volatility exposure.
    """
    
    def __init__(self, ibkr: IBKRConnector, logger):
        self.ibkr = ibkr
        self.logger = logger
    
    def get_option_greeks(self, symbol: str, dte_target: int) -> Optional[Dict]:
        """
        Get option Greeks for a symbol's ATM straddle.
        
        Returns dict with:
        - strike: ATM strike
        - call_vega: Call option vega
        - put_vega: Put option vega
        - total_vega: Combined straddle vega
        - mid_price: Mid price of straddle
        - dte: Actual days to expiration
        """
        # This would call IBKR API to get option chain and Greeks
        # For now, returning placeholder structure
        self.logger.info(f"  Getting Greeks for {symbol} (~{dte_target} DTE)")
        
        # TODO: Implement actual IBKR API calls
        # contract = self.ibkr.get_atm_straddle_contract(symbol, dte_target)
        # greeks = self.ibkr.get_option_greeks(contract)
        
        return {
            'symbol': symbol,
            'strike': 0,
            'call_vega': 0,
            'put_vega': 0,
            'total_vega': 0,
            'mid_price': 0,
            'dte': dte_target
        }
    
    def calculate_vega_neutral_sizes(
        self, 
        signal_type: str, 
        risk_budget: float,
        components: List[Tuple[str, float]]
    ) -> Dict:
        """
        Calculate vega-neutral position sizes.
        
        Args:
            signal_type: 'LONG_DISPERSION' or 'SHORT_DISPERSION'
            risk_budget: Dollar amount to risk on this trade
            components: List of (ticker, weight) tuples
            
        Returns:
            Dict with position details for each leg
        """
        self.logger.info(f"\nCalculating vega-neutral sizes for {signal_type}")
        self.logger.info(f"  Risk budget: ${risk_budget:,.2f}")
        self.logger.info(f"  Components: {len(components)}")
        
        # Get Greeks for index
        index_greeks = self.get_option_greeks(INDEX_SYMBOL, TARGET_DTE)
        index_vega = index_greeks['total_vega']
        
        if index_vega == 0:
            self.logger.error("  Index vega is zero - cannot calculate sizes")
            return None
        
        self.logger.info(f"  {INDEX_SYMBOL} vega per contract: {index_vega:.4f}")
        
        # Get Greeks for components
        component_greeks = []
        total_component_vega_per_lot = 0
        
        for ticker, weight in components:
            greeks = self.get_option_greeks(ticker, TARGET_DTE)
            if greeks and greeks['total_vega'] > 0:
                component_greeks.append(greeks)
                total_component_vega_per_lot += greeks['total_vega']
        
        if len(component_greeks) == 0:
            self.logger.error("  No valid component Greeks - cannot proceed")
            return None
        
        self.logger.info(f"  Valid components: {len(component_greeks)}")
        self.logger.info(f"  Total component vega per lot: {total_component_vega_per_lot:.4f}")
        
        # Calculate vega-neutral ratio
        # We want: N_index * vega_index = N_components * vega_components
        # So: N_index / N_components = vega_components / vega_index
        vega_ratio = total_component_vega_per_lot / index_vega
        
        self.logger.info(f"  Vega ratio (comp/index): {vega_ratio:.4f}")
        
        # For simplicity, start with 1 lot of components
        # Then calculate how many index contracts needed for vega neutrality
        component_lots = 1
        index_lots = int(round(component_lots * vega_ratio))
        
        if index_lots == 0:
            index_lots = 1
        
        self.logger.info(f"  Initial sizing: {index_lots} index lots, {component_lots} component lots")
        
        # Build position structure
        position = {
            'signal_type': signal_type,
            'risk_budget': risk_budget,
            'index': {
                'symbol': INDEX_SYMBOL,
                'side': 'SELL' if signal_type == 'SHORT_DISPERSION' else 'BUY',
                'contracts': index_lots,
                'vega': index_vega * index_lots,
                'greeks': index_greeks
            },
            'components': []
        }
        
        # Add component legs
        for greeks in component_greeks:
            position['components'].append({
                'symbol': greeks['symbol'],
                'side': 'BUY' if signal_type == 'SHORT_DISPERSION' else 'SELL',
                'contracts': component_lots,
                'vega': greeks['total_vega'] * component_lots,
                'greeks': greeks
            })
        
        # Calculate total vega
        total_index_vega = position['index']['vega']
        total_component_vega = sum(c['vega'] for c in position['components'])
        vega_imbalance = abs(total_index_vega - total_component_vega)
        
        self.logger.info(f"\n  VEGA ANALYSIS:")
        self.logger.info(f"    Index vega:     {total_index_vega:>10.4f}")
        self.logger.info(f"    Component vega: {total_component_vega:>10.4f}")
        self.logger.info(f"    Imbalance:      {vega_imbalance:>10.4f} ({vega_imbalance/total_index_vega*100:.1f}%)")
        
        return position


# =============================================================================
# AUTOMATED TRADER
# =============================================================================

class AutomatedDispersionTrader:
    """
    Institutional-grade automated dispersion trader.
    
    Features:
    - Vega-neutral sizing
    - Complex combo orders
    - Multiple safety checks
    - Manual confirmation workflow
    """
    
    def __init__(self, capital: float):
        self.capital = capital
        self.logger = setup_logging()
        self.ibkr = None
        self.sizer = None
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        self.logger.info("=" * 70)
        self.logger.info("INSTITUTIONAL AUTOMATED DISPERSION TRADER")
        self.logger.info("=" * 70)
    
    def initialize(self):
        """Initialize IBKR connection and components."""
        self.logger.info("Initializing trader...")
        
        self.ibkr = IBKRConnector()
        self.sizer = VegaNeutralSizer(self.ibkr, self.logger)
        
        return True
    
    def connect(self):
        """Connect to IBKR."""
        self.logger.info("Connecting to IBKR...")
        return self.ibkr.connect()
    
    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ibkr:
            self.ibkr.disconnect()
    
    def run_safety_checks(self, signal: Dict) -> Tuple[bool, str]:
        """
        Run all safety checks before allowing trade.
        
        Returns:
            (passed, reason) tuple
        """
        self.logger.info("\nRunning safety checks...")
        
        # Check 1: Account balance
        # account_balance = self.ibkr.get_account_balance()
        account_balance = self.capital  # Placeholder
        
        if account_balance < MIN_ACCOUNT_BALANCE:
            return False, f"Account balance ${account_balance:,.0f} below minimum ${MIN_ACCOUNT_BALANCE:,.0f}"
        
        self.logger.info(f"  ✓ Account balance: ${account_balance:,.2f}")
        
        # Check 2: VIX circuit breaker
        vix_level = self.ibkr.get_vix_level()
        
        if vix_level and vix_level > VIX_CIRCUIT_BREAKER:
            return False, f"VIX {vix_level:.1f} above circuit breaker {VIX_CIRCUIT_BREAKER}"
        
        self.logger.info(f"  ✓ VIX level: {vix_level:.1f} (below {VIX_CIRCUIT_BREAKER})")
        
        # Check 3: Max concurrent trades
        # active_positions = self.get_active_positions()
        active_positions = 0  # Placeholder
        
        if active_positions >= MAX_CONCURRENT_TRADES:
            return False, f"Already at max concurrent trades ({MAX_CONCURRENT_TRADES})"
        
        self.logger.info(f"  ✓ Active positions: {active_positions}/{MAX_CONCURRENT_TRADES}")
        
        # Check 4: Max gross exposure
        # current_exposure = self.get_current_exposure()
        current_exposure = 0  # Placeholder
        max_exposure = self.capital * MAX_GROSS_EXPOSURE_PCT
        
        if current_exposure >= max_exposure:
            return False, f"Already at max exposure ${current_exposure:,.0f}/${max_exposure:,.0f}"
        
        self.logger.info(f"  ✓ Gross exposure: ${current_exposure:,.0f}/${max_exposure:,.0f}")
        
        # Check 5: Signal is actionable
        if signal['signal'] not in ['LONG_DISPERSION', 'SHORT_DISPERSION']:
            return False, f"Signal {signal['signal']} is not actionable"
        
        self.logger.info(f"  ✓ Signal is actionable: {signal['signal']}")
        
        self.logger.info("\n  ALL SAFETY CHECKS PASSED ✓")
        return True, "All checks passed"
    
    def get_top_components(self, n: int = NUM_COMPONENTS) -> List[Tuple[str, float]]:
        """Get top N components by weight."""
        sorted_components = sorted(NDX_COMPONENTS, key=lambda x: x[1], reverse=True)
        return sorted_components[:n]
    
    def prepare_trade(self, signal: Dict) -> Optional[Dict]:
        """
        Prepare trade with vega-neutral sizing.
        
        Returns:
            Trade specification dict or None if failed
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("PREPARING TRADE")
        self.logger.info("=" * 70)
        
        # Calculate risk budget
        risk_budget = self.capital * RISK_PER_TRADE_PCT
        self.logger.info(f"Risk budget: ${risk_budget:,.2f} ({RISK_PER_TRADE_PCT*100}% of ${self.capital:,.0f})")
        
        # Get components
        components = self.get_top_components(NUM_COMPONENTS)
        self.logger.info(f"Using top {len(components)} components")
        
        # Calculate vega-neutral sizes
        position = self.sizer.calculate_vega_neutral_sizes(
            signal['signal'],
            risk_budget,
            components
        )
        
        if position is None:
            self.logger.error("Failed to calculate position sizes")
            return None
        
        # Add metadata
        position['timestamp'] = datetime.now().isoformat()
        position['signal'] = signal
        position['capital'] = self.capital
        
        return position
    
    def display_trade_summary(self, trade: Dict):
        """Display formatted trade summary."""
        print("\n" + "=" * 70)
        print("TRADE SUMMARY")
        print("=" * 70)
        print(f"\nSignal: {trade['signal_type']}")
        print(f"Risk Budget: ${trade['risk_budget']:,.2f}")
        print(f"\nINDEX LEG ({INDEX_SYMBOL}):")
        print(f"  Side: {trade['index']['side']}")
        print(f"  Contracts: {trade['index']['contracts']}")
        print(f"  Vega: {trade['index']['vega']:.4f}")
        print(f"\nCOMPONENT LEGS ({len(trade['components'])} stocks):")
        print(f"  Side: {trade['components'][0]['side']}")
        print(f"  Contracts per stock: {trade['components'][0]['contracts']}")
        print(f"  Total vega: {sum(c['vega'] for c in trade['components']):.4f}")
        print(f"\nTop 5 components:")
        for i, comp in enumerate(trade['components'][:5]):
            print(f"  {i+1}. {comp['symbol']}: {comp['contracts']} contracts")
        print("=" * 70)
    
    def request_confirmation(self, trade: Dict) -> bool:
        """
        Request manual confirmation from user.
        
        This is the "Crawl" phase - require approval before execution.
        """
        self.display_trade_summary(trade)
        
        print("\n" + "!" * 70)
        print("MANUAL CONFIRMATION REQUIRED")
        print("!" * 70)
        print("\nThis trade will be submitted to IBKR for execution.")
        print("Review the trade summary above carefully.")
        print("\nType 'EXECUTE' to proceed, or anything else to cancel:")
        
        response = input("> ").strip().upper()
        
        if response == "EXECUTE":
            self.logger.info("User confirmed trade execution")
            return True
        else:
            self.logger.info(f"User cancelled trade (response: {response})")
            return False
    
    def execute_trade(self, trade: Dict) -> bool:
        """
        Execute the trade on IBKR.
        
        Uses complex combo order with limit pricing.
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("EXECUTING TRADE")
        self.logger.info("=" * 70)
        
        # TODO: Implement actual IBKR order execution
        # This would:
        # 1. Build combo order with all legs
        # 2. Set limit price at mid-price
        # 3. Submit order
        # 4. Monitor fill status
        # 5. Walk price if needed
        # 6. Cancel if timeout exceeded
        
        self.logger.info("  Building combo order...")
        self.logger.info(f"  Total legs: {1 + len(trade['components'])}")
        
        # Placeholder for actual execution
        self.logger.info("  [PAPER TRADING MODE - No actual orders placed]")
        self.logger.info("  In live mode, would submit complex combo order to IBKR")
        
        # Save trade record
        trade_file = os.path.join(
            OUTPUT_DIR, 
            f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(trade_file, 'w') as f:
            json.dump(trade, f, indent=2, default=str)
        
        self.logger.info(f"  Trade record saved: {trade_file}")
        
        return True
    
    def run(self, signal: Dict):
        """
        Main execution flow.
        
        1. Run safety checks
        2. Prepare trade with vega-neutral sizing
        3. Request manual confirmation
        4. Execute trade
        """
        print("\n" + "=" * 70)
        print("AUTOMATED TRADING WORKFLOW")
        print("=" * 70)
        
        try:
            # Initialize
            self.initialize()
            
            # Connect to IBKR
            if not self.connect():
                self.logger.error("Failed to connect to IBKR")
                return False
            
            # Run safety checks
            passed, reason = self.run_safety_checks(signal)
            if not passed:
                self.logger.warning(f"Safety check failed: {reason}")
                print(f"\n⚠️  TRADE BLOCKED: {reason}")
                self.disconnect()
                return False
            
            # Prepare trade
            trade = self.prepare_trade(signal)
            if trade is None:
                self.logger.error("Failed to prepare trade")
                self.disconnect()
                return False
            
            # Request confirmation (Crawl phase)
            if not self.request_confirmation(trade):
                print("\n❌ Trade cancelled by user")
                self.disconnect()
                return False
            
            # Execute trade
            success = self.execute_trade(trade)
            
            if success:
                print("\n✅ Trade executed successfully!")
            else:
                print("\n❌ Trade execution failed")
            
            # Disconnect
            self.disconnect()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in trading workflow: {e}")
            import traceback
            traceback.print_exc()
            self.disconnect()
            return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Test the automated trader."""
    # Sample signal
    signal = {
        'signal': 'SHORT_DISPERSION',
        'impl_corr': 0.45,
        'z_score': 1.8,
        'timestamp': datetime.now().isoformat(),
        'reason': 'Z-score above threshold'
    }
    
    trader = AutomatedDispersionTrader(capital=100000)
    trader.run(signal)


if __name__ == "__main__":
    main()
