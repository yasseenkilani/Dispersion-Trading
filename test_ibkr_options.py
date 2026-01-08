"""
IBKR Options Data Test Script
=============================
Tests what option data and Greeks we can get from IBKR.

Run this with TWS open:
    python3 test_ibkr_options.py
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
from datetime import datetime, timedelta

class OptionDataTester(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}
        self.option_chains = {}
        self.greeks = {}
        self.next_req_id = 1
        self.connected = False
        
    def nextValidId(self, orderId):
        self.connected = True
        print(f"✓ Connected to IBKR (Order ID: {orderId})")
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2158, 2119]:  # Ignore connection messages
            print(f"  Error {errorCode}: {errorString}")
    
    # ===========================================
    # TEST 1: Get Option Chain Parameters
    # ===========================================
    def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId, 
                                          tradingClass, multiplier, expirations, strikes):
        """Receives option chain parameters"""
        print(f"\n  Option Chain for reqId {reqId}:")
        print(f"    Exchange: {exchange}")
        print(f"    Trading Class: {tradingClass}")
        print(f"    Multiplier: {multiplier}")
        print(f"    Expirations: {list(expirations)[:5]}... ({len(expirations)} total)")
        print(f"    Strikes: {sorted(list(strikes))[:10]}... ({len(strikes)} total)")
        
        self.option_chains[reqId] = {
            'exchange': exchange,
            'expirations': list(expirations),
            'strikes': sorted(list(strikes))
        }
    
    def securityDefinitionOptionParameterEnd(self, reqId):
        print(f"  Option chain request {reqId} complete")
    
    # ===========================================
    # TEST 2: Get Option Greeks (via Market Data)
    # ===========================================
    def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol, 
                              delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        """Receives option Greeks"""
        tick_names = {
            10: "Bid", 11: "Ask", 12: "Last", 13: "Model",
            53: "Bid", 54: "Ask", 55: "Last", 56: "Model"
        }
        tick_name = tick_names.get(tickType, f"Type{tickType}")
        
        if impliedVol is not None and impliedVol > 0:
            print(f"\n  Greeks for reqId {reqId} ({tick_name}):")
            print(f"    Implied Vol: {impliedVol*100:.2f}%")
            print(f"    Delta: {delta:.4f}" if delta else "    Delta: N/A")
            print(f"    Gamma: {gamma:.6f}" if gamma else "    Gamma: N/A")
            print(f"    Vega: {vega:.4f}" if vega else "    Vega: N/A")
            print(f"    Theta: {theta:.4f}" if theta else "    Theta: N/A")
            print(f"    Underlying Price: ${undPrice:.2f}" if undPrice else "    Underlying: N/A")
            
            self.greeks[reqId] = {
                'iv': impliedVol,
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta,
                'underlying': undPrice
            }
    
    def tickPrice(self, reqId, tickType, price, attrib):
        """Receives price data"""
        if tickType in [1, 2, 4]:  # Bid, Ask, Last
            tick_names = {1: "Bid", 2: "Ask", 4: "Last"}
            print(f"    {tick_names.get(tickType, tickType)}: ${price:.2f}")


def create_stock_contract(symbol):
    """Create a stock contract"""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def create_option_contract(symbol, strike, expiry, right="C"):
    """Create an option contract"""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "OPT"
    contract.exchange = "SMART"
    contract.currency = "USD"
    contract.strike = strike
    contract.lastTradeDateOrContractMonth = expiry  # Format: YYYYMMDD
    contract.right = right  # "C" for Call, "P" for Put
    contract.multiplier = "100"
    return contract


def main():
    print("=" * 60)
    print("IBKR OPTIONS DATA TEST")
    print("=" * 60)
    print("\nThis script tests what option data we can get from IBKR.")
    print("Make sure TWS/Gateway is running and logged in.\n")
    
    # Connect
    app = OptionDataTester()
    app.connect("127.0.0.1", 7497, clientId=99)  # 7497 for paper, 7496 for live
    
    # Start message thread
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    
    # Wait for connection
    timeout = 10
    while not app.connected and timeout > 0:
        time.sleep(1)
        timeout -= 1
    
    if not app.connected:
        print("✗ Failed to connect to IBKR")
        print("  Make sure TWS is running and API is enabled")
        return
    
    time.sleep(1)
    
    # ===========================================
    # TEST 1: Get Option Chain for QQQ
    # ===========================================
    print("\n" + "=" * 50)
    print("TEST 1: Getting QQQ Option Chain Parameters")
    print("=" * 50)
    
    qqq_contract = create_stock_contract("QQQ")
    app.reqSecDefOptParams(1, "QQQ", "", "STK", 320227571)  # QQQ conId
    time.sleep(3)
    
    # ===========================================
    # TEST 2: Get Option Chain for AAPL
    # ===========================================
    print("\n" + "=" * 50)
    print("TEST 2: Getting AAPL Option Chain Parameters")
    print("=" * 50)
    
    app.reqSecDefOptParams(2, "AAPL", "", "STK", 265598)  # AAPL conId
    time.sleep(3)
    
    # ===========================================
    # TEST 3: Get Greeks for a specific QQQ option
    # ===========================================
    print("\n" + "=" * 50)
    print("TEST 3: Getting Greeks for QQQ ATM Call Option")
    print("=" * 50)
    
    # Find next monthly expiry (3rd Friday)
    today = datetime.now()
    # Approximate ATM strike (you may need to adjust)
    atm_strike = 520  # Adjust based on current QQQ price
    
    # Find next month's expiry
    next_month = today.replace(day=1) + timedelta(days=32)
    next_month = next_month.replace(day=1)
    # Find 3rd Friday
    first_day = next_month.weekday()
    days_to_friday = (4 - first_day) % 7
    third_friday = next_month + timedelta(days=days_to_friday + 14)
    expiry = third_friday.strftime("%Y%m%d")
    
    print(f"  Looking for QQQ {atm_strike}C expiring {expiry}")
    
    qqq_option = create_option_contract("QQQ", atm_strike, expiry, "C")
    app.reqMktData(10, qqq_option, "106", False, False, [])  # 106 = option IV
    time.sleep(5)
    
    # ===========================================
    # TEST 4: Get Greeks for NVDA option
    # ===========================================
    print("\n" + "=" * 50)
    print("TEST 4: Getting Greeks for NVDA ATM Call Option")
    print("=" * 50)
    
    nvda_strike = 135  # Adjust based on current NVDA price
    print(f"  Looking for NVDA {nvda_strike}C expiring {expiry}")
    
    nvda_option = create_option_contract("NVDA", nvda_strike, expiry, "C")
    app.reqMktData(11, nvda_option, "106", False, False, [])
    time.sleep(5)
    
    # ===========================================
    # TEST 5: Get Greeks for AAPL option
    # ===========================================
    print("\n" + "=" * 50)
    print("TEST 5: Getting Greeks for AAPL ATM Call Option")
    print("=" * 50)
    
    aapl_strike = 250  # Adjust based on current AAPL price
    print(f"  Looking for AAPL {aapl_strike}C expiring {expiry}")
    
    aapl_option = create_option_contract("AAPL", aapl_strike, expiry, "C")
    app.reqMktData(12, aapl_option, "106", False, False, [])
    time.sleep(5)
    
    # ===========================================
    # SUMMARY
    # ===========================================
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n1. Option Chains Retrieved:")
    if app.option_chains:
        for req_id, chain in app.option_chains.items():
            print(f"   ✓ Request {req_id}: {len(chain['expirations'])} expirations, {len(chain['strikes'])} strikes")
    else:
        print("   ✗ No option chains retrieved")
    
    print("\n2. Greeks Retrieved:")
    if app.greeks:
        for req_id, greeks in app.greeks.items():
            print(f"   ✓ Request {req_id}: IV={greeks['iv']*100:.1f}%, Vega={greeks['vega']:.4f}")
    else:
        print("   ✗ No Greeks retrieved")
    
    print("\n" + "=" * 60)
    if app.greeks:
        print("SUCCESS! We CAN get option Greeks from IBKR.")
        print("Vega-neutral execution is POSSIBLE with current setup.")
    else:
        print("Greeks not available - may need OPRA subscription or different approach")
    print("=" * 60)
    
    # Disconnect
    app.disconnect()
    print("\n✓ Disconnected from IBKR")


if __name__ == "__main__":
    main()
