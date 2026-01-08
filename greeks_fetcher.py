"""
Greeks Fetcher Module
=====================
Fetches real-time option Greeks from IBKR for vega-neutral position sizing.

Features:
- Option chain retrieval
- ATM strike identification
- Real-time Greeks (IV, Delta, Gamma, Vega, Theta)
- Vega-neutral position size calculation
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    IBAPI_AVAILABLE = True
except ImportError:
    IBAPI_AVAILABLE = False
    print("Warning: ibapi not installed. Running in simulation mode.")

import threading


@dataclass
class OptionGreeks:
    """Container for option Greeks."""
    symbol: str
    strike: float
    expiry: str
    right: str  # "C" or "P"
    implied_vol: float
    delta: float
    gamma: float
    vega: float
    theta: float
    underlying_price: float


class GreeksFetcher:
    """
    Fetches option Greeks from IBKR.
    
    Usage:
        fetcher = GreeksFetcher()
        fetcher.connect_to_ibkr()
        greeks = fetcher.get_option_greeks("QQQ", 520, "20250117", "C")
        fetcher.disconnect()
    """
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize the Greeks fetcher."""
        self.logger = logger or logging.getLogger(__name__)
        self.app = None
        self.connected = False
        self.greeks_data = {}
        self.underlying_prices = {}
        self.option_chains = {}
        
    def connect_to_ibkr(self, host: str = "127.0.0.1", port: int = 7497, 
                        client_id: int = 51) -> bool:
        """
        Connect to IBKR TWS/Gateway.
        
        Args:
            host: IBKR host
            port: 7497 for paper, 7496 for live
            client_id: Unique client ID
            
        Returns:
            True if connected successfully
        """
        if not IBAPI_AVAILABLE:
            self.logger.warning("IBAPI not available, using simulation mode")
            return True
        
        try:
            self.app = IBKRGreeksApp(self)
            self.app.connect(host, port, client_id)
            
            # Start message processing thread
            api_thread = threading.Thread(target=self.app.run, daemon=True)
            api_thread.start()
            
            # Wait for connection
            timeout = 10
            start = time.time()
            while not self.app.isConnected() and time.time() - start < timeout:
                time.sleep(0.1)
            
            if self.app.isConnected():
                self.connected = True
                self.logger.info(f"Connected to IBKR on port {port}")
                time.sleep(1)  # Allow connection to stabilize
                return True
            else:
                self.logger.error("Failed to connect to IBKR")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IBKR."""
        if self.app and self.connected:
            self.app.disconnect()
            self.connected = False
            self.logger.info("Disconnected from IBKR")
    
    def get_underlying_price(self, symbol: str) -> Optional[float]:
        """Get current price of underlying."""
        if not self.connected:
            # Simulation mode - return approximate prices
            sim_prices = {
                'QQQ': 520, 'NVDA': 135, 'AAPL': 250, 'MSFT': 430,
                'AMZN': 225, 'META': 610, 'GOOGL': 195, 'GOOG': 195,
                'AVGO': 240, 'TSLA': 420, 'COST': 950, 'NFLX': 900
            }
            return sim_prices.get(symbol, 100)
        
        # Request market data
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        req_id = hash(symbol) % 10000
        self.underlying_prices[req_id] = None
        
        self.app.reqMktData(req_id, contract, "", False, False, [])
        
        # Wait for price
        timeout = 5
        start = time.time()
        while self.underlying_prices.get(req_id) is None and time.time() - start < timeout:
            time.sleep(0.1)
        
        self.app.cancelMktData(req_id)
        
        return self.underlying_prices.get(req_id)
    
    def get_atm_strike(self, symbol: str, price: float = None) -> float:
        """Get ATM strike for a symbol."""
        if price is None:
            price = self.get_underlying_price(symbol)
        
        if price is None:
            return 0
        
        # Round to nearest standard strike
        if price < 50:
            strike_interval = 1
        elif price < 200:
            strike_interval = 5
        else:
            strike_interval = 10
        
        return round(price / strike_interval) * strike_interval
    
    def get_next_monthly_expiry(self) -> str:
        """Get the next monthly option expiration date."""
        today = datetime.now()
        year, month = today.year, today.month
        
        # Find third Friday of current month
        first_day = datetime(year, month, 1)
        days_to_friday = (4 - first_day.weekday()) % 7
        third_friday = first_day + timedelta(days=days_to_friday + 14)
        
        # If third Friday is within 7 days, use next month
        if third_friday <= today + timedelta(days=7):
            month += 1
            if month > 12:
                month, year = 1, year + 1
            first_day = datetime(year, month, 1)
            days_to_friday = (4 - first_day.weekday()) % 7
            third_friday = first_day + timedelta(days=days_to_friday + 14)
        
        return third_friday.strftime("%Y%m%d")
    
    def get_option_greeks(self, symbol: str, strike: float = None, 
                          expiry: str = None, right: str = "C") -> Optional[OptionGreeks]:
        """
        Get Greeks for a specific option.
        
        Args:
            symbol: Underlying symbol
            strike: Option strike (None for ATM)
            expiry: Expiration date YYYYMMDD (None for next monthly)
            right: "C" for call, "P" for put
            
        Returns:
            OptionGreeks object or None
        """
        # Get underlying price
        price = self.get_underlying_price(symbol)
        
        # Default to ATM strike
        if strike is None:
            strike = self.get_atm_strike(symbol, price)
        
        # Default to next monthly expiry
        if expiry is None:
            expiry = self.get_next_monthly_expiry()
        
        if not self.connected:
            # Simulation mode - estimate Greeks
            return self._simulate_greeks(symbol, strike, expiry, right, price)
        
        # Request option Greeks from IBKR
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.strike = strike
        contract.lastTradeDateOrContractMonth = expiry
        contract.right = right
        contract.multiplier = "100"
        
        req_id = hash(f"{symbol}_{strike}_{expiry}_{right}") % 10000 + 10
        self.greeks_data[req_id] = None
        
        # Request market data with Greeks
        self.app.reqMktData(req_id, contract, "106", False, False, [])
        
        # Wait for Greeks
        timeout = 10
        start = time.time()
        while self.greeks_data.get(req_id) is None and time.time() - start < timeout:
            time.sleep(0.1)
        
        self.app.cancelMktData(req_id)
        
        greeks = self.greeks_data.get(req_id)
        if greeks:
            return OptionGreeks(
                symbol=symbol,
                strike=strike,
                expiry=expiry,
                right=right,
                implied_vol=greeks.get('impliedVol', 0),
                delta=greeks.get('delta', 0),
                gamma=greeks.get('gamma', 0),
                vega=greeks.get('vega', 0),
                theta=greeks.get('theta', 0),
                underlying_price=price or 0
            )
        
        return None
    
    def _simulate_greeks(self, symbol: str, strike: float, expiry: str, 
                         right: str, price: float) -> OptionGreeks:
        """Simulate Greeks for testing without IBKR connection."""
        # Estimate days to expiry
        try:
            exp_date = datetime.strptime(expiry, "%Y%m%d")
            dte = (exp_date - datetime.now()).days
        except:
            dte = 30
        
        # Rough IV estimates by symbol
        iv_estimates = {
            'QQQ': 0.20, 'NVDA': 0.55, 'AAPL': 0.25, 'MSFT': 0.25,
            'AMZN': 0.35, 'META': 0.40, 'GOOGL': 0.30, 'GOOG': 0.30,
            'AVGO': 0.40, 'TSLA': 0.60, 'COST': 0.20, 'NFLX': 0.45
        }
        iv = iv_estimates.get(symbol, 0.30)
        
        # Simplified Black-Scholes Greeks for ATM
        import math
        T = dte / 365.0
        sqrt_T = math.sqrt(T) if T > 0 else 0.01
        
        # ATM approximations
        delta = 0.5 if right == "C" else -0.5
        gamma = 0.4 / (price * iv * sqrt_T) if price > 0 and iv > 0 else 0
        vega = price * sqrt_T * 0.4 / 100  # Per 1% IV change
        theta = -price * iv * 0.4 / (2 * sqrt_T * 365) if sqrt_T > 0 else 0
        
        return OptionGreeks(
            symbol=symbol,
            strike=strike,
            expiry=expiry,
            right=right,
            implied_vol=iv,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            underlying_price=price or 0
        )
    
    def get_straddle_greeks(self, symbol: str, strike: float = None,
                            expiry: str = None) -> Dict:
        """
        Get combined Greeks for an ATM straddle.
        
        Returns:
            Dict with combined straddle Greeks
        """
        call_greeks = self.get_option_greeks(symbol, strike, expiry, "C")
        put_greeks = self.get_option_greeks(symbol, strike, expiry, "P")
        
        if not call_greeks or not put_greeks:
            return {}
        
        return {
            'symbol': symbol,
            'strike': call_greeks.strike,
            'expiry': call_greeks.expiry,
            'underlying_price': call_greeks.underlying_price,
            'straddle_delta': call_greeks.delta + put_greeks.delta,
            'straddle_gamma': call_greeks.gamma + put_greeks.gamma,
            'straddle_vega': call_greeks.vega + put_greeks.vega,
            'straddle_theta': call_greeks.theta + put_greeks.theta,
            'call_iv': call_greeks.implied_vol,
            'put_iv': put_greeks.implied_vol,
            'avg_iv': (call_greeks.implied_vol + put_greeks.implied_vol) / 2
        }
    
    def get_all_vegas(self, index_symbol: str, 
                      component_symbols: List[str]) -> Dict:
        """
        Get vega for index and all components.
        
        Args:
            index_symbol: Index symbol (e.g., "QQQ")
            component_symbols: List of component symbols
            
        Returns:
            Dict with all vega data
        """
        self.logger.info(f"Fetching vegas for {index_symbol} and {len(component_symbols)} components...")
        
        expiry = self.get_next_monthly_expiry()
        
        # Get index straddle Greeks
        index_greeks = self.get_straddle_greeks(index_symbol, expiry=expiry)
        
        # Get component straddle Greeks
        component_vegas = {}
        for symbol in component_symbols:
            greeks = self.get_straddle_greeks(symbol, expiry=expiry)
            if greeks:
                component_vegas[symbol] = {
                    'vega': greeks['straddle_vega'],
                    'details': greeks
                }
        
        return {
            'index_symbol': index_symbol,
            'index_vega': index_greeks.get('straddle_vega', 0),
            'index_details': index_greeks,
            'component_vegas': component_vegas,
            'expiry': expiry,
            'timestamp': datetime.now().isoformat()
        }


class VegaNeutralSizer:
    """
    Calculates vega-neutral position sizes for dispersion trades.
    
    Goal: Total component vega exposure = Index vega exposure
    """
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize the sizer."""
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_position_sizes(self, index_vega: float, 
                                  component_vegas: Dict[str, Dict],
                                  capital: float,
                                  max_position_pct: float = 0.02,
                                  num_components: int = 30) -> Dict:
        """
        Calculate vega-neutral position sizes.
        
        Args:
            index_vega: Vega of index straddle
            component_vegas: Dict of {symbol: {'vega': float, ...}}
            capital: Total trading capital
            max_position_pct: Maximum position size as % of capital
            num_components: Number of components to trade
            
        Returns:
            Dict with position sizes for index and each component
        """
        max_position = capital * max_position_pct
        
        if not index_vega or not component_vegas:
            self.logger.warning("Missing vega data, using equal weighting")
            return self._equal_weight_sizing(capital, max_position_pct, num_components)
        
        # Get top components by vega
        sorted_components = sorted(
            component_vegas.items(),
            key=lambda x: x[1].get('vega', 0),
            reverse=True
        )[:num_components]
        
        # Calculate total component vega
        total_component_vega = sum(v.get('vega', 0) for _, v in sorted_components)
        
        if total_component_vega <= 0:
            self.logger.warning("Total component vega is zero, using equal weighting")
            return self._equal_weight_sizing(capital, max_position_pct, num_components)
        
        # Vega-neutral sizing:
        # We want: index_contracts * index_vega = sum(component_contracts * component_vega)
        # 
        # Approach: Allocate capital proportionally to inverse vega
        # Higher vega stocks get smaller positions (fewer contracts)
        
        # Calculate inverse vega weights
        inverse_vegas = {}
        for symbol, data in sorted_components:
            vega = data.get('vega', 0)
            if vega > 0:
                inverse_vegas[symbol] = 1 / vega
        
        total_inverse_vega = sum(inverse_vegas.values())
        
        if total_inverse_vega <= 0:
            return self._equal_weight_sizing(capital, max_position_pct, num_components)
        
        # Allocate component capital
        component_allocation = max_position * 0.5  # 50% to components
        index_allocation = max_position * 0.5      # 50% to index
        
        # Calculate per-component allocation (inverse vega weighted)
        component_positions = {}
        total_allocated_vega = 0
        
        for symbol, inv_vega in inverse_vegas.items():
            weight = inv_vega / total_inverse_vega
            allocation = component_allocation * weight
            
            component_vega = component_vegas[symbol].get('vega', 0)
            total_allocated_vega += component_vega * (allocation / 100)  # Rough contract estimate
            
            component_positions[symbol] = {
                'allocation': allocation,
                'weight': weight,
                'vega': component_vega
            }
        
        # Adjust index allocation to match component vega
        # This is the key vega-neutral adjustment
        if index_vega > 0:
            vega_ratio = total_allocated_vega / index_vega
            adjusted_index_allocation = index_allocation * vega_ratio
            
            # Cap at max position
            adjusted_index_allocation = min(adjusted_index_allocation, max_position * 0.6)
        else:
            adjusted_index_allocation = index_allocation
        
        # Check if vega-neutral
        is_vega_neutral = abs(vega_ratio - 1.0) < 0.2 if index_vega > 0 else False
        
        self.logger.info(f"Vega ratio: {vega_ratio:.2f} (1.0 = perfectly neutral)")
        self.logger.info(f"Is vega-neutral: {is_vega_neutral}")
        
        return {
            'index': {
                'symbol': 'QQQ',
                'allocation': adjusted_index_allocation,
                'vega': index_vega
            },
            'components': component_positions,
            'summary': {
                'is_vega_neutral': is_vega_neutral,
                'vega_ratio': vega_ratio,
                'total_allocation': adjusted_index_allocation + component_allocation,
                'index_allocation': adjusted_index_allocation,
                'component_allocation': component_allocation,
                'num_components': len(component_positions)
            }
        }
    
    def _equal_weight_sizing(self, capital: float, max_position_pct: float,
                              num_components: int) -> Dict:
        """Fallback to equal weight sizing."""
        max_position = capital * max_position_pct
        index_allocation = max_position * 0.5
        component_allocation = max_position * 0.5
        per_component = component_allocation / num_components
        
        return {
            'index': {
                'symbol': 'QQQ',
                'allocation': index_allocation
            },
            'components': {
                f'COMP_{i}': {'allocation': per_component}
                for i in range(num_components)
            },
            'summary': {
                'is_vega_neutral': False,
                'total_allocation': max_position,
                'index_allocation': index_allocation,
                'component_allocation': component_allocation,
                'num_components': num_components
            }
        }


# IBKR API Wrapper (only if ibapi is available)
if IBAPI_AVAILABLE:
    class IBKRGreeksApp(EWrapper, EClient):
        """IBKR API wrapper for Greeks fetching."""
        
        def __init__(self, fetcher: GreeksFetcher):
            EClient.__init__(self, self)
            self.fetcher = fetcher
        
        def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
            if errorCode not in [2104, 2106, 2158, 10167]:  # Ignore info messages
                self.fetcher.logger.debug(f"IBKR Error {errorCode}: {errorString}")
        
        def tickPrice(self, reqId, tickType, price, attrib):
            if tickType == 4:  # Last price
                self.fetcher.underlying_prices[reqId] = price
        
        def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol,
                                   delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
            if tickType in [10, 11, 12, 13]:  # Model-based Greeks
                self.fetcher.greeks_data[reqId] = {
                    'impliedVol': impliedVol if impliedVol > 0 else None,
                    'delta': delta,
                    'gamma': gamma,
                    'vega': vega,
                    'theta': theta,
                    'undPrice': undPrice
                }


# ===========================================
# Test
# ===========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("GREEKS FETCHER TEST (Simulation Mode)")
    print("=" * 60)
    
    fetcher = GreeksFetcher(logger)
    
    # Test without IBKR connection (simulation)
    print("\nTesting simulated Greeks...")
    
    # Get QQQ straddle Greeks
    qqq_greeks = fetcher.get_straddle_greeks("QQQ")
    print(f"\nQQQ Straddle Greeks:")
    print(f"  Strike: ${qqq_greeks['strike']}")
    print(f"  Vega: {qqq_greeks['straddle_vega']:.4f}")
    print(f"  Delta: {qqq_greeks['straddle_delta']:.4f}")
    
    # Get component Greeks
    components = ['NVDA', 'AAPL', 'MSFT']
    print(f"\nComponent Greeks:")
    for symbol in components:
        greeks = fetcher.get_straddle_greeks(symbol)
        print(f"  {symbol}: Vega={greeks['straddle_vega']:.4f}")
    
    # Test vega-neutral sizing
    print("\n" + "=" * 60)
    print("VEGA-NEUTRAL SIZING TEST")
    print("=" * 60)
    
    sizer = VegaNeutralSizer(logger)
    
    # Get all vegas
    all_vegas = fetcher.get_all_vegas("QQQ", components)
    
    # Calculate position sizes
    sizes = sizer.calculate_position_sizes(
        index_vega=all_vegas['index_vega'],
        component_vegas=all_vegas['component_vegas'],
        capital=100000,
        max_position_pct=0.02,
        num_components=3
    )
    
    print(f"\nPosition Sizes:")
    print(f"  Index allocation: ${sizes['index']['allocation']:,.2f}")
    print(f"  Component allocations:")
    for symbol, data in sizes['components'].items():
        print(f"    {symbol}: ${data['allocation']:,.2f} (weight: {data.get('weight', 0):.2%})")
    print(f"\n  Is vega-neutral: {sizes['summary']['is_vega_neutral']}")
    print(f"  Vega ratio: {sizes['summary'].get('vega_ratio', 'N/A')}")
