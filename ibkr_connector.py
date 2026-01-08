"""
IBKR Connector for Dispersion Trading
=====================================

This module handles connection to Interactive Brokers TWS/Gateway
and fetches real-time implied volatility data for QQQ and components.

UPDATED: Now fetches IV from OPTION contracts (not stocks) for accurate data.
Uses ATM straddle IV for each component.

Requirements:
- IBKR TWS or IB Gateway running
- Paper trading or live account
- Market data subscriptions for options (OPRA)

Usage:
    from ibkr_connector import IBKRConnector
    
    connector = IBKRConnector()
    connector.connect()
    iv_data = connector.get_component_ivs()
    connector.disconnect()
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import queue

# =============================================================================
# CONFIGURATION
# =============================================================================

# IBKR Connection Settings
TWS_HOST = "127.0.0.1"
TWS_PORT = 7497  # 7497 for paper trading, 7496 for live
CLIENT_ID = 1

# Data Mode: Set to True to use delayed data (free), False for real-time (requires subscription)
USE_DELAYED_DATA = True

# INDEX CONFIGURATION
INDEX_SYMBOL = "QQQ"  # ETF tracking NDX
INDEX_TYPE = "STK"    # QQQ is a stock/ETF, not an index
VXN_SYMBOL = "VXN"    # NASDAQ Volatility Index (backup for IV)

# NDX Components (top holdings by weight)
NDX_COMPONENTS = [
    ("NVDA", 0.1213), ("AAPL", 0.1147), ("GOOGL", 0.1055), ("GOOG", 0.1055),
    ("MSFT", 0.0999), ("AMZN", 0.0674), ("META", 0.0462), ("AVGO", 0.0454),
    ("TSLA", 0.0448), ("NFLX", 0.0121), ("ASML", 0.0120), ("COST", 0.0108),
    ("AMD", 0.0096), ("CSCO", 0.0088), ("MU", 0.0076), ("TMUS", 0.0062),
    ("AMAT", 0.0059), ("PEP", 0.0059), ("LRCX", 0.0058), ("ISRG", 0.0056),
    ("QCOM", 0.0054), ("INTU", 0.0052), ("INTC", 0.0051), ("BKNG", 0.0050),
    ("AMGN", 0.0050), ("TXN", 0.0046), ("KLAC", 0.0046), ("PDD", 0.0044),
    ("GILD", 0.0042), ("ADBE", 0.0042), ("ADI", 0.0039), ("ARM", 0.0037),
    ("PANW", 0.0037), ("HON", 0.0036), ("CRWD", 0.0035), ("VRTX", 0.0033),
    ("CEG", 0.0032), ("ADP", 0.0030), ("CMCSA", 0.0029), ("MELI", 0.0028),
    ("DASH", 0.0028), ("SBUX", 0.0028), ("CDNS", 0.0025), ("SNPS", 0.0024),
    ("MAR", 0.0023), ("ABNB", 0.0023), ("ORLY", 0.0023), ("REGN", 0.0022),
    ("CTAS", 0.0021), ("WBD", 0.0021),
]

# =============================================================================
# IBKR WRAPPER CLASS
# =============================================================================

class IBKRWrapper(EWrapper):
    """Handles callbacks from IBKR API."""
    
    def __init__(self):
        EWrapper.__init__(self)
        self.iv_data = {}
        self.price_data = {}
        self.data_received = threading.Event()
        self.error_queue = queue.Queue()
        self.next_order_id = None
        self.contract_details = {}
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Handle errors from IBKR."""
        # Ignore common info messages
        if errorCode in [2104, 2106, 2158, 10167]:
            return
        error_msg = f"Error {errorCode}: {errorString}"
        if reqId != -1:
            error_msg = f"ReqId {reqId} - {error_msg}"
        # Only print actual errors, not subscription warnings
        if errorCode not in [10089, 10090, 354]:
            print(f"  IBKR: {error_msg}")
        self.error_queue.put((reqId, errorCode, errorString))
    
    def nextValidId(self, orderId):
        """Receive next valid order ID."""
        self.next_order_id = orderId
    
    def tickPrice(self, reqId, tickType, price, attrib):
        """Handle price tick data."""
        # LAST=4, CLOSE=9, DELAYED_LAST=68, DELAYED_CLOSE=75
        if tickType in [4, 9, 68, 75] and price > 0:
            self.price_data[reqId] = price
            self.data_received.set()
    
    def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol, 
                               delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        """Handle option computation data including implied volatility."""
        # tickType 10 = BID_OPTION, 11 = ASK_OPTION, 12 = LAST_OPTION, 13 = MODEL_OPTION
        if impliedVol is not None and impliedVol > 0:
            # Convert to percentage
            iv_pct = impliedVol * 100
            self.iv_data[reqId] = {
                'iv': iv_pct,
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta,
                'undPrice': undPrice
            }
            self.data_received.set()
    
    def tickGeneric(self, reqId, tickType, value):
        """Handle generic tick data."""
        # tickType 24 = OPTION_IMPLIED_VOL
        if tickType == 24 and value > 0:
            self.iv_data[reqId] = {'iv': value * 100}
            self.data_received.set()
    
    def contractDetails(self, reqId, contractDetails):
        """Handle contract details response."""
        if reqId not in self.contract_details:
            self.contract_details[reqId] = []
        self.contract_details[reqId].append(contractDetails)
    
    def contractDetailsEnd(self, reqId):
        """Contract details request completed."""
        self.data_received.set()


# =============================================================================
# IBKR CLIENT CLASS
# =============================================================================

class IBKRClient(EClient):
    """IBKR API client."""
    
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)


# =============================================================================
# MAIN CONNECTOR CLASS
# =============================================================================

class IBKRConnector:
    """
    Main connector class for IBKR dispersion trading.
    
    UPDATED: Now fetches IV from OPTION contracts for accurate data.
    Uses ATM straddle Greeks from actual options.
    """
    
    def __init__(self, host=TWS_HOST, port=TWS_PORT, client_id=CLIENT_ID):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.wrapper = IBKRWrapper()
        self.client = IBKRClient(self.wrapper)
        self.connected = False
        self.req_id_counter = 1
        self._underlying_prices = {}
        self._next_expiry = None
        
    def connect(self):
        """Connect to IBKR TWS/Gateway."""
        print(f"Connecting to IBKR at {self.host}:{self.port}...")
        
        self.client.connect(self.host, self.port, self.client_id)
        
        # Start message processing thread
        self.api_thread = threading.Thread(target=self.client.run, daemon=True)
        self.api_thread.start()
        
        # Wait for connection
        time.sleep(2)
        
        if self.client.isConnected():
            self.connected = True
            
            # Switch to delayed data if configured
            if USE_DELAYED_DATA:
                print("  Switching to delayed market data...")
                self.client.reqMarketDataType(3)
                time.sleep(0.5)
            
            print("Connected to IBKR")
            return True
        else:
            print("Failed to connect to IBKR")
            return False
    
    def disconnect(self):
        """Disconnect from IBKR."""
        if self.connected:
            self.client.disconnect()
            self.connected = False
            print("Disconnected from IBKR")
    
    def _get_next_req_id(self):
        """Get next request ID."""
        req_id = self.req_id_counter
        self.req_id_counter += 1
        return req_id
    
    def _get_next_monthly_expiry(self):
        """
        Get the monthly option expiration closest to 30 DTE.
        
        For dispersion trading, we want options with ~21-45 DTE.
        Uses the 3rd Friday of the month (standard monthly expiration).
        
        FIXED: Now properly targets 30 DTE by checking multiple months
        and selecting the expiry closest to 30 days out.
        """
        if self._next_expiry:
            return self._next_expiry
            
        today = datetime.now()
        target_dte = 30  # Target days to expiration
        min_dte = 21     # Minimum acceptable DTE
        max_dte = 45     # Maximum acceptable DTE
        
        # Collect potential expiries from current and next 2 months
        candidates = []
        
        for month_offset in range(3):  # Current month + next 2 months
            year = today.year
            month = today.month + month_offset
            
            # Handle year rollover
            while month > 12:
                month -= 12
                year += 1
            
            # Find third Friday of this month
            first_day = datetime(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_until_friday)
            third_friday = first_friday + timedelta(days=14)
            
            # Calculate DTE
            dte = (third_friday - today).days
            
            # Only consider if in acceptable range and in the future
            if dte >= min_dte and dte <= max_dte:
                candidates.append((third_friday, dte))
        
        # If no candidates in ideal range, expand search
        if not candidates:
            for month_offset in range(3):
                year = today.year
                month = today.month + month_offset
                
                while month > 12:
                    month -= 12
                    year += 1
                
                first_day = datetime(year, month, 1)
                days_until_friday = (4 - first_day.weekday()) % 7
                first_friday = first_day + timedelta(days=days_until_friday)
                third_friday = first_friday + timedelta(days=14)
                
                dte = (third_friday - today).days
                
                # Accept any future expiry with at least 14 DTE
                if dte >= 14:
                    candidates.append((third_friday, dte))
        
        if not candidates:
            # Fallback: use next month's third Friday
            month = today.month + 1
            year = today.year
            if month > 12:
                month = 1
                year += 1
            first_day = datetime(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_until_friday)
            third_friday = first_friday + timedelta(days=14)
            candidates.append((third_friday, (third_friday - today).days))
        
        # Select the expiry closest to target DTE (30 days)
        best_expiry, best_dte = min(candidates, key=lambda x: abs(x[1] - target_dte))
        
        self._next_expiry = best_expiry.strftime("%Y%m%d")
        print(f"  Option expiry: {best_expiry.strftime('%b %d, %Y')} ({best_dte} DTE)")
        return self._next_expiry
    
    def _get_underlying_price(self, symbol, timeout=5):
        """Get current price of underlying stock."""
        if symbol in self._underlying_prices:
            return self._underlying_prices[symbol]
        
        req_id = self._get_next_req_id()
        
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        if USE_DELAYED_DATA:
            self.client.reqMarketDataType(3)
        
        self.wrapper.data_received.clear()
        self.wrapper.price_data[req_id] = None
        
        self.client.reqMktData(req_id, contract, "", False, False, [])
        
        # Wait for price
        start = time.time()
        while self.wrapper.price_data.get(req_id) is None and time.time() - start < timeout:
            time.sleep(0.1)
        
        self.client.cancelMktData(req_id)
        
        price = self.wrapper.price_data.get(req_id)
        if price:
            self._underlying_prices[symbol] = price
        return price
    
    def _get_atm_strike(self, symbol, price=None):
        """Get ATM strike for a symbol."""
        if price is None:
            price = self._get_underlying_price(symbol)
        
        if not price:
            return None
        
        # Round to nearest standard strike interval
        if price < 25:
            interval = 1
        elif price < 50:
            interval = 2.5
        elif price < 200:
            interval = 5
        else:
            interval = 10
        
        return round(price / interval) * interval
    
    def create_option_contract(self, symbol, strike, expiry, right="C"):
        """Create an option contract."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = float(strike)
        contract.right = right
        contract.multiplier = "100"
        return contract
    
    def get_option_iv(self, symbol, timeout=8):
        """
        Get implied volatility from ATM option for a symbol.
        
        This is the CORRECT way to get IV - from actual options, not stocks.
        """
        # Get underlying price
        price = self._get_underlying_price(symbol, timeout=3)
        if not price:
            return None
        
        # Get ATM strike
        strike = self._get_atm_strike(symbol, price)
        if not strike:
            return None
        
        # Get next monthly expiry
        expiry = self._get_next_monthly_expiry()
        
        # Request IV from ATM call option
        req_id = self._get_next_req_id()
        contract = self.create_option_contract(symbol, strike, expiry, "C")
        
        if USE_DELAYED_DATA:
            self.client.reqMarketDataType(3)
        
        self.wrapper.data_received.clear()
        self.wrapper.iv_data[req_id] = None
        
        # Request market data - this will trigger tickOptionComputation
        self.client.reqMktData(req_id, contract, "", False, False, [])
        
        # Wait for IV data
        start = time.time()
        while self.wrapper.iv_data.get(req_id) is None and time.time() - start < timeout:
            time.sleep(0.1)
        
        self.client.cancelMktData(req_id)
        
        # Return IV if available
        iv_data = self.wrapper.iv_data.get(req_id)
        if iv_data and iv_data.get('iv'):
            return iv_data['iv']
        
        return None
    
    def get_stock_iv(self, symbol, timeout=8):
        """
        Get implied volatility for a stock using ATM option.
        
        UPDATED: Now fetches from option contract, not stock.
        """
        return self.get_option_iv(symbol, timeout)
    
    def get_qqq_iv(self, timeout=10):
        """
        Get implied volatility for QQQ.
        
        First tries to get from QQQ options, falls back to VXN.
        """
        print(f"  Fetching {INDEX_SYMBOL} IV...")
        
        iv = self.get_option_iv(INDEX_SYMBOL, timeout)
        
        if iv is not None:
            print(f"  {INDEX_SYMBOL} IV = {iv:.2f}%")
            return iv
        
        # Fallback to VXN if QQQ option IV not available
        print(f"  {INDEX_SYMBOL} option IV not available, using VXN as proxy...")
        return self.get_vxn_level(timeout)
    
    def get_index_iv(self, symbol="QQQ", timeout=10):
        """Get implied volatility for the index proxy (QQQ)."""
        if symbol in ["QQQ", "NDX"]:
            return self.get_qqq_iv(timeout)
        return self.get_option_iv(symbol, timeout)
    
    def get_vxn_level(self, timeout=10):
        """Get VXN (NASDAQ Volatility Index) level as a proxy for index IV."""
        req_id = self._get_next_req_id()
        
        contract = Contract()
        contract.symbol = "VXN"
        contract.secType = "IND"
        contract.exchange = "CBOE"
        contract.currency = "USD"
        
        if USE_DELAYED_DATA:
            self.client.reqMarketDataType(3)
        
        self.wrapper.data_received.clear()
        self.wrapper.price_data[req_id] = None
        
        self.client.reqMktData(req_id, contract, "", False, False, [])
        
        time.sleep(timeout)
        self.client.cancelMktData(req_id)
        
        if req_id in self.wrapper.price_data and self.wrapper.price_data[req_id]:
            vxn_level = self.wrapper.price_data[req_id]
            print(f"  VXN level: {vxn_level:.2f}%")
            return vxn_level
        
        return None
    
    def get_vix_level(self, timeout=10):
        """Get VIX level as a backup proxy for index implied volatility."""
        req_id = self._get_next_req_id()
        
        contract = Contract()
        contract.symbol = "VIX"
        contract.secType = "IND"
        contract.exchange = "CBOE"
        contract.currency = "USD"
        
        if USE_DELAYED_DATA:
            self.client.reqMarketDataType(3)
        
        self.wrapper.data_received.clear()
        self.wrapper.price_data[req_id] = None
        
        self.client.reqMktData(req_id, contract, "", False, False, [])
        
        time.sleep(timeout)
        self.client.cancelMktData(req_id)
        
        if req_id in self.wrapper.price_data and self.wrapper.price_data[req_id]:
            vix_level = self.wrapper.price_data[req_id]
            print(f"  VIX level: {vix_level:.2f}%")
            return vix_level
        
        return None
    
    def get_all_component_ivs(self, components=None, timeout_per_stock=8):
        """
        Get implied volatility for all components using ATM options.
        
        UPDATED: Now fetches from option contracts for accurate IV data.
        """
        if components is None:
            components = NDX_COMPONENTS
        
        # Reset expiry cache to recalculate
        self._next_expiry = None
        
        print(f"\nFetching IV for {len(components)} components (from options)...")
        
        iv_results = {}
        
        for i, (symbol, weight) in enumerate(components):
            print(f"  [{i+1}/{len(components)}] {symbol}...", end=" ", flush=True)
            
            iv = self.get_option_iv(symbol, timeout=timeout_per_stock)
            
            if iv is not None:
                iv_results[symbol] = iv
                print(f"IV = {iv:.2f}%")
            else:
                print("No data")
            
            time.sleep(0.3)  # Rate limiting
        
        print(f"\nGot IV for {len(iv_results)}/{len(components)} components")
        return iv_results
    
    def get_dispersion_data(self):
        """
        Get all data needed for dispersion signal calculation.
        
        Returns:
            Dict with 'index_iv', 'component_ivs', 'weights', 'timestamp'
        """
        print("\n" + "=" * 50)
        print("FETCHING DISPERSION DATA (Option-Based IV)")
        print("=" * 50)
        
        # Get QQQ IV
        print(f"\n1. Fetching {INDEX_SYMBOL} IV...")
        index_iv = self.get_qqq_iv()
        if index_iv:
            print(f"   {INDEX_SYMBOL} IV = {index_iv:.2f}%")
        else:
            print(f"   Warning: Could not get {INDEX_SYMBOL} IV")
        
        # Get component IVs
        print("\n2. Fetching component IVs (from ATM options)...")
        component_ivs = self.get_all_component_ivs()
        
        # Create weights dict
        weights = {symbol: weight for symbol, weight in NDX_COMPONENTS}
        
        return {
            'index_iv': index_iv,
            'ndx_iv': index_iv,
            'qqq_iv': index_iv,
            'component_ivs': component_ivs,
            'weights': weights,
            'timestamp': datetime.now(),
            'index_symbol': INDEX_SYMBOL
        }


# =============================================================================
# STANDALONE TEST
# =============================================================================

def test_connection():
    """Test IBKR connection and option-based IV fetching."""
    print("=" * 60)
    print("IBKR CONNECTOR TEST (Option-Based IV)")
    print("=" * 60)
    print("\nMake sure TWS or IB Gateway is running!")
    print(f"Connecting to {TWS_HOST}:{TWS_PORT}...")
    
    connector = IBKRConnector()
    
    if connector.connect():
        print("\n--- Testing Option-Based IV Fetch ---")
        
        # Show what expiry we're using
        print("\nCalculating option expiry...")
        expiry = connector._get_next_monthly_expiry()
        
        # Test underlying price
        print("\nTesting underlying price fetch...")
        aapl_price = connector._get_underlying_price("AAPL")
        if aapl_price:
            print(f"  AAPL price = ${aapl_price:.2f}")
            print(f"  ATM strike = ${connector._get_atm_strike('AAPL', aapl_price)}")
        
        # Test option IV for a few stocks
        test_symbols = ["AAPL", "NVDA", "MSFT", "CDNS"]
        for symbol in test_symbols:
            print(f"\nTesting {symbol} option IV fetch...")
            iv = connector.get_option_iv(symbol)
            if iv:
                print(f"  {symbol} IV = {iv:.2f}%")
            else:
                print(f"  Could not get {symbol} IV")
        
        # Test QQQ
        print("\nTesting QQQ IV fetch...")
        qqq_iv = connector.get_qqq_iv()
        if qqq_iv:
            print(f"  QQQ IV = {qqq_iv:.2f}%")
        
        # Test VXN
        print("\nTesting VXN fetch...")
        vxn = connector.get_vxn_level()
        if vxn:
            print(f"  VXN = {vxn:.2f}%")
        
        connector.disconnect()
    else:
        print("\nConnection failed. Check that:")
        print("1. TWS or IB Gateway is running")
        print("2. API connections are enabled in settings")
        print("3. Port number is correct (7497 for paper, 7496 for live)")


if __name__ == "__main__":
    test_connection()
