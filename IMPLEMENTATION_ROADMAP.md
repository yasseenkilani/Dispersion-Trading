# Institutional Automated Trading Implementation Roadmap

## Overview

This document outlines the implementation plan for the institutional-grade automated dispersion trading system, following professional quant desk best practices.

## Current Status: Framework Complete ✓

The core framework has been implemented in `automated_trader.py` with:
- ✓ Vega-neutral position sizing architecture
- ✓ Safety checks and circuit breakers
- ✓ Manual confirmation workflow (Crawl phase)
- ✓ Logging and audit trail

## What Needs to be Completed

### Phase 1: IBKR API Integration (CRITICAL)

The following methods in `ibkr_connector.py` need to be implemented:

#### 1.1 Option Chain Retrieval
```python
def get_option_chain(self, symbol: str, expiration_date: str):
    """
    Get full option chain for a symbol and expiration.
    Returns list of contracts with strikes and Greeks.
    """
    # TODO: Implement using reqContractDetails and reqMktData
```

#### 1.2 ATM Straddle Selection
```python
def get_atm_straddle_contract(self, symbol: str, target_dte: int):
    """
    Find ATM straddle contracts closest to target DTE.
    Returns (call_contract, put_contract, actual_dte).
    """
    # TODO: 
    # 1. Get current stock/ETF price
    # 2. Find expiration closest to target_dte
    # 3. Find strike closest to current price
    # 4. Return call and put contracts at that strike
```

#### 1.3 Option Greeks Retrieval
```python
def get_option_greeks(self, contract):
    """
    Get Greeks (delta, gamma, vega, theta) for an option contract.
    Returns dict with all Greeks.
    """
    # TODO: Use reqMktData with genericTickList="106" for Greeks
```

#### 1.4 Complex Combo Order Submission
```python
def submit_combo_order(self, legs: List[Dict], limit_price: float):
    """
    Submit a multi-leg combo order.
    
    Args:
        legs: List of dicts with {contract, action, quantity}
        limit_price: Net limit price for entire combo
        
    Returns:
        order_id
    """
    # TODO: 
    # 1. Create BAG contract with all legs
    # 2. Create LMT order with limit_price
    # 3. Submit via placeOrder
    # 4. Return order ID for tracking
```

#### 1.5 Order Status Monitoring
```python
def monitor_order_status(self, order_id: int, timeout: int = 30):
    """
    Monitor order fill status with timeout.
    Returns ('FILLED', 'PARTIAL', 'CANCELLED', 'TIMEOUT').
    """
    # TODO: 
    # 1. Subscribe to order status updates
    # 2. Wait for fill or timeout
    # 3. Return status
```

### Phase 2: Vega Calculator Enhancement

The `VegaNeutralSizer` class needs real Greeks data:

```python
def get_option_greeks(self, symbol: str, dte_target: int):
    """Currently returns placeholder data - needs real implementation."""
    
    # TODO:
    # 1. Call ibkr.get_atm_straddle_contract(symbol, dte_target)
    # 2. Call ibkr.get_option_greeks() for call and put
    # 3. Calculate total straddle vega
    # 4. Get mid-price from bid/ask
    # 5. Return complete Greeks dict
```

### Phase 3: Order Execution Logic

Implement the "walk the book" algorithm in `execute_trade()`:

```python
def execute_trade(self, trade: Dict) -> bool:
    """
    TODO: Implement full execution logic:
    
    1. Build combo order from trade specification
    2. Get current mid-price for all legs
    3. Submit order at mid-price
    4. Wait 5 seconds
    5. If not filled:
       - Cancel order
       - Walk price by $0.01 towards ask
       - Resubmit
    6. Repeat until filled or max slippage exceeded
    7. Return success/failure
    """
```

### Phase 4: Position Management

Create position tracking system:

```python
def get_active_positions(self) -> List[Dict]:
    """Query IBKR for current open positions."""
    # TODO: Use reqPositions() to get all positions
    
def get_current_exposure(self) -> float:
    """Calculate total dollar exposure across all positions."""
    # TODO: Sum notional value of all open positions
```

## Testing Checklist

### Unit Tests
- [ ] Test vega-neutral sizing with mock Greeks
- [ ] Test safety checks with various scenarios
- [ ] Test order building logic
- [ ] Test price walking algorithm

### Integration Tests (Paper Trading)
- [ ] Connect to IBKR paper account
- [ ] Retrieve real option chains
- [ ] Get real Greeks data
- [ ] Submit test combo order (1 contract)
- [ ] Monitor order fill
- [ ] Verify position appears in account

### System Tests
- [ ] Run full signal → trade → execution workflow
- [ ] Test VIX circuit breaker
- [ ] Test max position limits
- [ ] Test manual confirmation flow
- [ ] Test error handling (bad connection, invalid contracts, etc.)

## Deployment Phases

### Crawl (Months 1-2): Manual Confirmation
- System generates trade specification
- User reviews and types "EXECUTE" to confirm
- System submits order to IBKR
- **Goal**: Verify all plumbing works correctly

### Walk (Months 3-4): Auto-Execute with Alert
- System auto-submits orders
- User receives immediate SMS/email alert
- User has 60 seconds to cancel via web interface
- **Goal**: Build confidence in automation

### Run (Months 5+): Full Automation
- System runs completely autonomously
- User receives end-of-day summary
- **Goal**: Hands-off operation

## Risk Management

### Pre-Trade Checks
- ✓ Account balance > $50,000
- ✓ VIX < 30
- ✓ Active positions < 3
- ✓ Gross exposure < 50% of capital

### Intra-Trade Monitoring
- [ ] Monitor order fill status
- [ ] Track slippage vs. expected
- [ ] Alert on abnormal fills

### Post-Trade Reconciliation
- [ ] Verify all legs filled
- [ ] Confirm vega neutrality achieved
- [ ] Log actual vs. expected costs

## Documentation Needed

- [ ] API integration guide
- [ ] Error handling procedures
- [ ] Emergency shutdown procedure
- [ ] Daily operations checklist
- [ ] Troubleshooting guide

## Estimated Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| IBKR API Integration | 1-2 weeks | Working option chain and Greeks retrieval |
| Vega Calculator | 3-5 days | Real vega-neutral sizing |
| Order Execution | 1 week | Complex combo orders working |
| Testing | 2-3 weeks | All tests passing |
| Paper Trading | 1-3 months | Verified in live market |
| **TOTAL** | **2-4 months** | Production-ready system |

## Next Steps

1. **Immediate**: Study IBKR API documentation for:
   - Option chain requests
   - Greeks retrieval
   - BAG orders (combo orders)
   - Order status callbacks

2. **Week 1**: Implement option chain and Greeks retrieval

3. **Week 2**: Implement combo order submission

4. **Week 3**: Build order monitoring and price walking

5. **Week 4**: Integration testing with paper account

6. **Month 2+**: Extended paper trading to verify reliability

## Support Resources

- IBKR API Documentation: https://interactivebrokers.github.io/tws-api/
- Python API Guide: https://interactivebrokers.github.io/tws-api/python_api.html
- Sample Code: https://github.com/InteractiveBrokers/tws-api-public

## Notes

- The framework is complete and follows institutional standards
- The missing piece is IBKR API integration (not trivial, but well-documented)
- Paper trading for 1-3 months is **mandatory** before live trading
- Start with small position sizes (1 contract) during testing
