"""
Automated Dispersion Trader v2
==============================
Institutional-grade automated trading with optional vega-neutral sizing.

Features:
- Real-time Greeks fetching from IBKR
- Vega-neutral position sizing (optional)
- Safety checks and circuit breakers
- Manual confirmation workflow (Crawl phase)
- Comprehensive logging and audit trail

Usage:
    from automated_trader_v2 import AutomatedDispersionTrader
    
    trader = AutomatedDispersionTrader(capital=100000, use_vega_neutral=True)
    trader.run(signal)
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import our modules
try:
    from greeks_fetcher import GreeksFetcher, VegaNeutralSizer
    from ibkr_connector import IBKRConnector
except ImportError:
    print("Warning: Some modules not found. Running in standalone mode.")


class TradingMode(Enum):
    """Trading automation levels."""
    CRAWL = "crawl"      # Manual confirmation required
    WALK = "walk"        # Auto-execute with cancel window
    RUN = "run"          # Fully autonomous


class SignalType(Enum):
    """Trading signal types."""
    LONG_DISPERSION = "LONG_DISPERSION"
    SHORT_DISPERSION = "SHORT_DISPERSION"
    NO_TRADE = "NO_TRADE"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class TradeOrder:
    """Represents a single leg of a dispersion trade."""
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: int
    order_type: str  # "STRADDLE"
    strike: float
    expiry: str
    estimated_price: float
    vega: float


@dataclass
class DispersionTrade:
    """Complete dispersion trade structure."""
    trade_id: str
    signal_type: SignalType
    timestamp: datetime
    index_order: TradeOrder
    component_orders: List[TradeOrder]
    total_vega_exposure: float
    is_vega_neutral: bool
    estimated_cost: float
    status: str  # "PENDING", "CONFIRMED", "EXECUTED", "CANCELLED"


class AutomatedDispersionTrader:
    """
    Automated dispersion trader with vega-neutral sizing capability.
    
    This class handles:
    1. Fetching real-time Greeks from IBKR
    2. Calculating vega-neutral position sizes (optional)
    3. Building trade orders
    4. Executing trades with safety checks
    """
    
    # Configuration
    DEFAULT_CONFIG = {
        'index_symbol': 'QQQ',
        'num_components': 30,
        'position_size_pct': 0.02,  # 2% of capital per trade
        'max_positions': 3,
        'max_gross_exposure': 0.50,  # 50% of capital
        'vix_circuit_breaker': 50,   # Block trades if VIX > 50
        'min_account_balance': 50000,
        'holding_period_days': 5,
        'trading_mode': TradingMode.CRAWL,
    }
    
    # Top 30 NDX components by weight
    TOP_COMPONENTS = [
        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'META', 'GOOGL', 'GOOG', 'AVGO', 
        'TSLA', 'COST', 'NFLX', 'TMUS', 'ASML', 'AMD', 'PEP', 'CSCO',
        'LIN', 'ADBE', 'TXN', 'QCOM', 'ISRG', 'INTU', 'CMCSA', 'AMGN',
        'HON', 'AMAT', 'BKNG', 'VRTX', 'ADP', 'GILD'
    ]
    
    def __init__(self, capital: float = 100000, use_vega_neutral: bool = False,
                 config: Dict = None, logger: logging.Logger = None):
        """
        Initialize the automated trader.
        
        Args:
            capital: Trading capital
            use_vega_neutral: Whether to use vega-neutral sizing
            config: Optional configuration overrides
            logger: Optional logger instance
        """
        self.capital = capital
        self.use_vega_neutral = use_vega_neutral
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        # Setup logging
        self.logger = logger or self._setup_logging()
        
        # Initialize components
        self.greeks_fetcher = None
        self.vega_sizer = VegaNeutralSizer(self.logger) if use_vega_neutral else None
        
        # State tracking
        self.open_positions: List[DispersionTrade] = []
        self.trade_history: List[DispersionTrade] = []
        self.current_vix = None
        
        self.logger.info(f"Initialized AutomatedDispersionTrader")
        self.logger.info(f"  Capital: ${capital:,.0f}")
        self.logger.info(f"  Vega-neutral: {use_vega_neutral}")
        self.logger.info(f"  Trading mode: {self.config['trading_mode'].value}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the trader."""
        logger = logging.getLogger('AutomatedTrader')
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler
        log_file = os.path.join(log_dir, f'automated_trader_{datetime.now():%Y%m%d}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    # ===========================================
    # Safety Checks
    # ===========================================
    
    def run_safety_checks(self) -> Tuple[bool, str]:
        """
        Run all safety checks before trading.
        
        Returns:
            Tuple of (passed: bool, message: str)
        """
        self.logger.info("Running safety checks...")
        
        # Check 1: Account balance
        if self.capital < self.config['min_account_balance']:
            return False, f"Account balance ${self.capital:,.0f} below minimum ${self.config['min_account_balance']:,.0f}"
        
        # Check 2: VIX circuit breaker
        if self.current_vix and self.current_vix > self.config['vix_circuit_breaker']:
            return False, f"VIX {self.current_vix:.1f} above circuit breaker {self.config['vix_circuit_breaker']}"
        
        # Check 3: Maximum positions
        if len(self.open_positions) >= self.config['max_positions']:
            return False, f"Maximum positions ({self.config['max_positions']}) reached"
        
        # Check 4: Gross exposure
        current_exposure = sum(t.estimated_cost for t in self.open_positions)
        max_exposure = self.capital * self.config['max_gross_exposure']
        if current_exposure >= max_exposure:
            return False, f"Gross exposure ${current_exposure:,.0f} at maximum ${max_exposure:,.0f}"
        
        self.logger.info("✓ All safety checks passed")
        return True, "All checks passed"
    
    # ===========================================
    # Greeks and Vega-Neutral Sizing
    # ===========================================
    
    def connect_to_ibkr(self, port: int = 7497) -> bool:
        """Connect to IBKR for Greeks data."""
        try:
            self.greeks_fetcher = GreeksFetcher(self.logger)
            connected = self.greeks_fetcher.connect_to_ibkr(port=port, client_id=51)
            
            if connected:
                self.logger.info("✓ Connected to IBKR for Greeks")
            else:
                self.logger.error("✗ Failed to connect to IBKR")
            
            return connected
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def fetch_greeks_data(self) -> Optional[Dict]:
        """
        Fetch Greeks for index and components.
        
        Returns:
            Dict with vega data for all instruments
        """
        if not self.greeks_fetcher:
            self.logger.warning("Greeks fetcher not initialized")
            return None
        
        self.logger.info(f"Fetching Greeks for {self.config['index_symbol']} and {self.config['num_components']} components...")
        
        try:
            greeks_data = self.greeks_fetcher.get_all_vegas(
                index_symbol=self.config['index_symbol'],
                component_symbols=self.TOP_COMPONENTS[:self.config['num_components']]
            )
            
            self.logger.info(f"✓ Got Greeks for {len(greeks_data.get('component_vegas', {}))} components")
            return greeks_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Greeks: {e}")
            return None
    
    def calculate_position_sizes(self, greeks_data: Dict) -> Dict:
        """
        Calculate position sizes (vega-neutral or standard).
        
        Args:
            greeks_data: Greeks data from fetch_greeks_data()
            
        Returns:
            Dict with position sizes for each instrument
        """
        if self.use_vega_neutral and self.vega_sizer and greeks_data:
            self.logger.info("Calculating vega-neutral position sizes...")
            
            position_sizes = self.vega_sizer.calculate_position_sizes(
                index_vega=greeks_data['index_vega'],
                component_vegas=greeks_data['component_vegas'],
                capital=self.capital,
                max_position_pct=self.config['position_size_pct'],
                num_components=self.config['num_components']
            )
            
            if position_sizes.get('summary', {}).get('is_vega_neutral'):
                self.logger.info("✓ Position sizes are vega-neutral")
            else:
                self.logger.warning("⚠ Position sizes are NOT perfectly vega-neutral")
            
            return position_sizes
        else:
            # Standard 50/50 sizing
            self.logger.info("Using standard 50/50 position sizing...")
            
            max_position = self.capital * self.config['position_size_pct']
            index_allocation = max_position * 0.5
            component_allocation = max_position * 0.5
            per_component = component_allocation / self.config['num_components']
            
            return {
                'index': {
                    'symbol': self.config['index_symbol'],
                    'allocation': index_allocation
                },
                'components': {
                    symbol: {'allocation': per_component}
                    for symbol in self.TOP_COMPONENTS[:self.config['num_components']]
                },
                'summary': {
                    'is_vega_neutral': False,
                    'total_allocation': max_position
                }
            }
    
    # ===========================================
    # Trade Building
    # ===========================================
    
    def build_trade(self, signal: Dict, position_sizes: Dict, 
                    greeks_data: Optional[Dict] = None) -> DispersionTrade:
        """
        Build a complete dispersion trade structure.
        
        Args:
            signal: Trading signal from signal generator
            position_sizes: Position sizes from calculate_position_sizes()
            greeks_data: Optional Greeks data for strike/expiry selection
            
        Returns:
            DispersionTrade object
        """
        signal_type = SignalType(signal.get('signal', 'NO_TRADE'))
        
        # Determine trade direction
        if signal_type == SignalType.SHORT_DISPERSION:
            index_action = "SELL"
            component_action = "BUY"
        else:  # LONG_DISPERSION
            index_action = "BUY"
            component_action = "SELL"
        
        # Get expiry (next monthly)
        if greeks_data and 'expiry' in greeks_data:
            expiry = greeks_data['expiry']
        else:
            # Calculate next monthly expiry
            today = datetime.now()
            year, month = today.year, today.month
            first_day = datetime(year, month, 1)
            days_to_friday = (4 - first_day.weekday()) % 7
            third_friday = first_day + timedelta(days=days_to_friday + 14)
            if third_friday <= today + timedelta(days=7):
                month += 1
                if month > 12:
                    month, year = 1, year + 1
                first_day = datetime(year, month, 1)
                days_to_friday = (4 - first_day.weekday()) % 7
                third_friday = first_day + timedelta(days=days_to_friday + 14)
            expiry = third_friday.strftime("%Y%m%d")
        
        # Build index order
        index_details = greeks_data.get('index_details', {}) if greeks_data else {}
        index_order = TradeOrder(
            symbol=self.config['index_symbol'],
            action=index_action,
            quantity=1,  # Will be calculated based on allocation
            order_type="STRADDLE",
            strike=index_details.get('strike', 0),
            expiry=expiry,
            estimated_price=position_sizes['index']['allocation'],
            vega=index_details.get('straddle_vega', 0)
        )
        
        # Build component orders
        component_orders = []
        total_component_vega = 0
        
        for symbol, size_data in position_sizes.get('components', {}).items():
            comp_details = {}
            if greeks_data and 'component_vegas' in greeks_data:
                comp_data = greeks_data['component_vegas'].get(symbol, {})
                comp_details = comp_data.get('details', {})
            
            order = TradeOrder(
                symbol=symbol,
                action=component_action,
                quantity=1,
                order_type="STRADDLE",
                strike=comp_details.get('strike', 0),
                expiry=expiry,
                estimated_price=size_data['allocation'],
                vega=comp_details.get('straddle_vega', 0) or size_data.get('vega', 0)
            )
            component_orders.append(order)
            total_component_vega += order.vega
        
        # Create trade object
        trade = DispersionTrade(
            trade_id=f"DSP_{datetime.now():%Y%m%d_%H%M%S}",
            signal_type=signal_type,
            timestamp=datetime.now(),
            index_order=index_order,
            component_orders=component_orders,
            total_vega_exposure=abs(index_order.vega - total_component_vega),
            is_vega_neutral=self.use_vega_neutral,
            estimated_cost=position_sizes['summary']['total_allocation'],
            status="PENDING"
        )
        
        return trade
    
    # ===========================================
    # Trade Display and Confirmation
    # ===========================================
    
    def display_trade_summary(self, trade: DispersionTrade):
        """Display a summary of the proposed trade."""
        print("\n" + "=" * 70)
        print("PROPOSED DISPERSION TRADE")
        print("=" * 70)
        
        print(f"\nTrade ID: {trade.trade_id}")
        print(f"Signal: {trade.signal_type.value}")
        print(f"Timestamp: {trade.timestamp}")
        print(f"Vega-Neutral: {'Yes' if trade.is_vega_neutral else 'No'}")
        
        print(f"\n{'─' * 70}")
        print("INDEX LEG:")
        print(f"{'─' * 70}")
        idx = trade.index_order
        print(f"  {idx.action} {idx.symbol} STRADDLE")
        print(f"    Strike: ${idx.strike:.0f}" if idx.strike else "    Strike: ATM")
        print(f"    Expiry: {idx.expiry}")
        print(f"    Allocation: ${idx.estimated_price:,.2f}")
        print(f"    Vega: {idx.vega:.4f}" if idx.vega else "    Vega: TBD")
        
        print(f"\n{'─' * 70}")
        print(f"COMPONENT LEGS ({len(trade.component_orders)} positions):")
        print(f"{'─' * 70}")
        
        for i, order in enumerate(trade.component_orders[:10]):  # Show first 10
            print(f"  {order.action} {order.symbol:5} STRADDLE - ${order.estimated_price:,.2f}")
        
        if len(trade.component_orders) > 10:
            print(f"  ... and {len(trade.component_orders) - 10} more")
        
        print(f"\n{'─' * 70}")
        print("SUMMARY:")
        print(f"{'─' * 70}")
        print(f"  Total Allocation: ${trade.estimated_cost:,.2f}")
        print(f"  Net Vega Exposure: {trade.total_vega_exposure:.4f}")
        print(f"  Components: {len(trade.component_orders)}")
        print("=" * 70)
    
    def request_confirmation(self, trade: DispersionTrade) -> bool:
        """
        Request manual confirmation for the trade (Crawl mode).
        
        Returns:
            True if confirmed, False if rejected
        """
        if self.config['trading_mode'] != TradingMode.CRAWL:
            return True
        
        print("\n" + "!" * 70)
        print("MANUAL CONFIRMATION REQUIRED")
        print("!" * 70)
        print("\nType 'EXECUTE' to confirm this trade, or 'CANCEL' to abort:")
        
        try:
            response = input("> ").strip().upper()
            
            if response == "EXECUTE":
                self.logger.info(f"Trade {trade.trade_id} CONFIRMED by user")
                return True
            else:
                self.logger.info(f"Trade {trade.trade_id} CANCELLED by user")
                return False
                
        except (EOFError, KeyboardInterrupt):
            self.logger.info(f"Trade {trade.trade_id} CANCELLED (interrupted)")
            return False
    
    # ===========================================
    # Trade Execution
    # ===========================================
    
    def execute_trade(self, trade: DispersionTrade) -> bool:
        """
        Execute the dispersion trade.
        
        Note: This is a placeholder for actual IBKR order submission.
        In production, this would submit combo orders to IBKR.
        
        Returns:
            True if executed successfully
        """
        self.logger.info(f"Executing trade {trade.trade_id}...")
        
        # TODO: Implement actual IBKR order submission
        # This would involve:
        # 1. Creating Contract objects for each leg
        # 2. Building a BAG (combo) order
        # 3. Submitting the order
        # 4. Monitoring fill status
        
        # For now, just log the trade
        trade.status = "EXECUTED"
        self.open_positions.append(trade)
        
        self.logger.info(f"✓ Trade {trade.trade_id} executed (paper)")
        
        # Save trade to file
        self._save_trade(trade)
        
        return True
    
    def _save_trade(self, trade: DispersionTrade):
        """Save trade details to file."""
        trades_dir = os.path.join(os.path.dirname(__file__), 'positions')
        os.makedirs(trades_dir, exist_ok=True)
        
        trade_file = os.path.join(trades_dir, f'{trade.trade_id}.json')
        
        trade_data = {
            'trade_id': trade.trade_id,
            'signal_type': trade.signal_type.value,
            'timestamp': trade.timestamp.isoformat(),
            'index_order': {
                'symbol': trade.index_order.symbol,
                'action': trade.index_order.action,
                'strike': trade.index_order.strike,
                'expiry': trade.index_order.expiry,
                'allocation': trade.index_order.estimated_price,
                'vega': trade.index_order.vega
            },
            'component_orders': [
                {
                    'symbol': o.symbol,
                    'action': o.action,
                    'strike': o.strike,
                    'expiry': o.expiry,
                    'allocation': o.estimated_price,
                    'vega': o.vega
                }
                for o in trade.component_orders
            ],
            'total_vega_exposure': trade.total_vega_exposure,
            'is_vega_neutral': trade.is_vega_neutral,
            'estimated_cost': trade.estimated_cost,
            'status': trade.status
        }
        
        with open(trade_file, 'w') as f:
            json.dump(trade_data, f, indent=2)
        
        self.logger.info(f"Trade saved to {trade_file}")
    
    # ===========================================
    # Main Entry Point
    # ===========================================
    
    def run(self, signal: Dict, auto_connect: bool = True) -> bool:
        """
        Main entry point for automated trading.
        
        Args:
            signal: Trading signal from signal generator
            auto_connect: Whether to auto-connect to IBKR
            
        Returns:
            True if trade was executed
        """
        self.logger.info("=" * 60)
        self.logger.info("AUTOMATED DISPERSION TRADER")
        self.logger.info("=" * 60)
        
        # Check signal
        signal_type = signal.get('signal', 'NO_TRADE')
        if signal_type in ['NO_TRADE', 'NO_SIGNAL']:
            self.logger.info(f"No actionable signal: {signal_type}")
            return False
        
        self.logger.info(f"Processing signal: {signal_type}")
        self.logger.info(f"  Implied Correlation: {signal.get('impl_corr', 'N/A')}")
        self.logger.info(f"  Z-Score: {signal.get('z_score', 'N/A')}")
        
        # Get VIX for safety check
        self.current_vix = signal.get('vix')
        
        # Run safety checks
        passed, message = self.run_safety_checks()
        if not passed:
            self.logger.warning(f"Safety check failed: {message}")
            return False
        
        # Connect to IBKR if needed
        greeks_data = None
        if self.use_vega_neutral and auto_connect:
            if self.connect_to_ibkr():
                greeks_data = self.fetch_greeks_data()
        
        # Calculate position sizes
        position_sizes = self.calculate_position_sizes(greeks_data)
        
        # Build trade
        trade = self.build_trade(signal, position_sizes, greeks_data)
        
        # Display summary
        self.display_trade_summary(trade)
        
        # Request confirmation (Crawl mode)
        if not self.request_confirmation(trade):
            trade.status = "CANCELLED"
            return False
        
        # Execute trade
        success = self.execute_trade(trade)
        
        # Disconnect from IBKR
        if self.greeks_fetcher:
            try:
                self.greeks_fetcher.disconnect()
            except:
                pass
        
        return success


# ===========================================
# Convenience Functions
# ===========================================

def run_automated_trade(signal: Dict, capital: float = 100000, 
                        use_vega_neutral: bool = False) -> bool:
    """
    Convenience function to run an automated trade.
    
    Args:
        signal: Trading signal dict
        capital: Trading capital
        use_vega_neutral: Whether to use vega-neutral sizing
        
    Returns:
        True if trade was executed
    """
    trader = AutomatedDispersionTrader(
        capital=capital,
        use_vega_neutral=use_vega_neutral
    )
    return trader.run(signal)


# ===========================================
# Test
# ===========================================

if __name__ == "__main__":
    # Test with a sample signal
    test_signal = {
        'signal': 'SHORT_DISPERSION',
        'impl_corr': 0.45,
        'z_score': 2.1,
        'vix': 18.5,
        'timestamp': datetime.now().isoformat()
    }
    
    print("\n" + "=" * 70)
    print("AUTOMATED TRADER TEST")
    print("=" * 70)
    print("\nTesting with sample SHORT_DISPERSION signal...")
    print("Vega-neutral: OFF (baseline mode)")
    print("\n")
    
    # Test baseline mode
    trader = AutomatedDispersionTrader(
        capital=100000,
        use_vega_neutral=False
    )
    
    # Run without IBKR connection (offline test)
    trader.run(test_signal, auto_connect=False)
