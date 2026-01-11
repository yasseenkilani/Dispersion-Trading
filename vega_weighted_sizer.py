"""
Vega-Weighted Position Sizer
============================

Combines Kelly criterion sizing with vega-neutral weighting for dispersion trades.

Position Sizing Flow:
1. Calculate total position size using Kelly criterion + confidence scaling
2. Split allocation: 50% index, 50% components
3. Apply vega-weighting to component allocations
4. Adjust index allocation to achieve vega neutrality

This ensures:
- Risk-adjusted position sizes (Kelly)
- Vega-neutral exposure (no directional vol bet)
- Proper hedging between index and components
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from risk_manager import RiskManager
from greeks_fetcher import GreeksFetcher, VegaNeutralSizer, OptionGreeks

logger = logging.getLogger(__name__)

# Module availability flag
VEGA_SIZER_AVAILABLE = True


@dataclass
class VegaWeightedPosition:
    """Container for a vega-weighted position."""
    symbol: str
    allocation: float
    contracts: int
    vega_per_contract: float
    total_vega: float
    weight: float
    side: str  # 'BUY' or 'SELL'


class VegaWeightedKellySizer:
    """
    Combines Kelly criterion with vega-neutral weighting.
    
    Strategy:
    - Kelly determines TOTAL position size
    - Vega weighting determines ALLOCATION between index and components
    """
    
    def __init__(self, 
                 base_capital: float = 100000,
                 kelly_fraction: float = 0.5,
                 base_win_rate: float = 0.669,
                 num_components: int = 30,
                 logger: logging.Logger = None):
        """
        Initialize the vega-weighted Kelly sizer.
        
        Args:
            base_capital: Starting capital
            kelly_fraction: Fraction of Kelly to use (0.5 = half Kelly)
            base_win_rate: Historical win rate for Kelly calculation
            num_components: Number of components to trade
            logger: Logger instance
        """
        self.base_capital = base_capital
        self.kelly_fraction = kelly_fraction
        self.base_win_rate = base_win_rate
        self.num_components = num_components
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize risk manager for Kelly calculations
        self.risk_manager = RiskManager(
            base_capital=base_capital,
            kelly_fraction=kelly_fraction,
            base_win_rate=base_win_rate
        )
        
        # Initialize Greeks fetcher
        self.greeks_fetcher = GreeksFetcher(self.logger)
        
        # Cache for vega data
        self.vega_cache = {}
        self.cache_timestamp = None
        self.cache_ttl = 300  # 5 minutes
    
    def _is_cache_valid(self) -> bool:
        """Check if vega cache is still valid."""
        if not self.cache_timestamp:
            return False
        elapsed = (datetime.now() - self.cache_timestamp).total_seconds()
        return elapsed < self.cache_ttl
    
    def fetch_vegas(self, index_symbol: str, component_symbols: List[str],
                    use_ibkr: bool = False) -> Dict:
        """
        Fetch vega data for index and components.
        
        Args:
            index_symbol: Index symbol (e.g., 'QQQ')
            component_symbols: List of component symbols
            use_ibkr: Whether to connect to IBKR for live data
            
        Returns:
            Dict with vega data
        """
        # Check cache
        if self._is_cache_valid():
            self.logger.info("Using cached vega data")
            return self.vega_cache
        
        # Connect to IBKR if requested
        if use_ibkr:
            connected = self.greeks_fetcher.connect_to_ibkr()
            if not connected:
                self.logger.warning("Could not connect to IBKR, using simulation")
        
        # Fetch vegas
        vega_data = self.greeks_fetcher.get_all_vegas(index_symbol, component_symbols)
        
        # Disconnect if connected
        if use_ibkr:
            self.greeks_fetcher.disconnect()
        
        # Update cache
        self.vega_cache = vega_data
        self.cache_timestamp = datetime.now()
        
        return vega_data
    
    def calculate_kelly_position(self, 
                                  current_capital: float,
                                  ml_probability: float) -> Dict:
        """
        Calculate Kelly-based total position size.
        
        Args:
            current_capital: Current portfolio capital
            ml_probability: ML model's probability
            
        Returns:
            Dict with Kelly position sizing info
        """
        # Use Kelly if kelly_fraction > 0, otherwise use fixed sizing
        use_kelly = self.risk_manager.kelly_fraction > 0
        return self.risk_manager.get_position_size(
            current_capital=current_capital,
            ml_probability=ml_probability,
            use_kelly=use_kelly
        )
    
    def calculate_vega_weighted_positions(self,
                                           signal_type: str,
                                           current_capital: float,
                                           ml_probability: float,
                                           index_symbol: str = 'QQQ',
                                           component_symbols: List[str] = None,
                                           use_ibkr: bool = False) -> Dict:
        """
        Calculate vega-weighted position sizes using Kelly criterion.
        
        Args:
            signal_type: 'SHORT_DISPERSION' or 'LONG_DISPERSION'
            current_capital: Current portfolio capital
            ml_probability: ML model's probability
            index_symbol: Index symbol
            component_symbols: List of component symbols
            use_ibkr: Whether to use live IBKR data
            
        Returns:
            Dict with complete position sizing
        """
        # Step 1: Calculate Kelly-based total position size
        kelly_sizing = self.calculate_kelly_position(current_capital, ml_probability)
        
        if not kelly_sizing['can_trade']:
            return {
                'can_trade': False,
                'reason': kelly_sizing['reason'],
                'kelly_sizing': kelly_sizing
            }
        
        total_position = kelly_sizing['position_size']
        
        # Step 2: Get default components if not provided
        if component_symbols is None:
            from ibkr_connector import NDX_COMPONENTS
            component_symbols = [c[0] for c in sorted(NDX_COMPONENTS, key=lambda x: x[1], reverse=True)[:self.num_components]]
        
        # Step 3: Fetch vega data
        vega_data = self.fetch_vegas(index_symbol, component_symbols, use_ibkr)
        
        index_vega = vega_data.get('index_vega', 0)
        component_vegas = vega_data.get('component_vegas', {})
        
        # Step 4: Calculate vega-weighted allocations
        positions = self._calculate_vega_allocations(
            signal_type=signal_type,
            total_position=total_position,
            index_vega=index_vega,
            component_vegas=component_vegas
        )
        
        # Step 5: Add Kelly info to result
        positions['kelly_sizing'] = kelly_sizing
        positions['ml_probability'] = ml_probability
        positions['current_capital'] = current_capital
        positions['can_trade'] = True
        
        return positions
    
    def _calculate_vega_allocations(self,
                                     signal_type: str,
                                     total_position: float,
                                     index_vega: float,
                                     component_vegas: Dict) -> Dict:
        """
        Calculate vega-weighted allocations.
        
        For SHORT_DISPERSION:
        - SELL index straddle (short vega on index)
        - BUY component straddles (long vega on components)
        
        For LONG_DISPERSION:
        - BUY index straddle (long vega on index)
        - SELL component straddles (short vega on components)
        """
        # Determine sides based on signal
        if signal_type == 'SHORT_DISPERSION':
            index_side = 'SELL'
            component_side = 'BUY'
        else:
            index_side = 'BUY'
            component_side = 'SELL'
        
        # Base allocation: 50% index, 50% components
        base_index_allocation = total_position * 0.5
        base_component_allocation = total_position * 0.5
        
        # If no vega data, use equal weighting
        if not index_vega or not component_vegas:
            return self._equal_weight_allocation(
                signal_type, total_position, index_side, component_side
            )
        
        # Calculate inverse-vega weights for components
        # Higher vega = smaller position (to balance vega exposure)
        inverse_vegas = {}
        for symbol, data in component_vegas.items():
            vega = data.get('vega', 0)
            if vega > 0:
                inverse_vegas[symbol] = 1 / vega
        
        total_inverse_vega = sum(inverse_vegas.values())
        
        if total_inverse_vega <= 0:
            return self._equal_weight_allocation(
                signal_type, total_position, index_side, component_side
            )
        
        # Calculate component allocations
        component_positions = {}
        total_component_vega_exposure = 0
        
        for symbol, inv_vega in inverse_vegas.items():
            weight = inv_vega / total_inverse_vega
            allocation = base_component_allocation * weight
            
            component_vega = component_vegas[symbol].get('vega', 0)
            
            # Estimate contracts (rough: allocation / (100 * avg_option_price))
            # Using vega as proxy for option price sensitivity
            estimated_contracts = max(1, int(allocation / 500))  # Rough estimate
            vega_exposure = component_vega * estimated_contracts
            total_component_vega_exposure += vega_exposure
            
            component_positions[symbol] = {
                'allocation': allocation,
                'weight': weight,
                'vega_per_contract': component_vega,
                'estimated_contracts': estimated_contracts,
                'vega_exposure': vega_exposure,
                'side': component_side
            }
        
        # Adjust index allocation to achieve vega neutrality
        if index_vega > 0 and total_component_vega_exposure > 0:
            # We want: index_vega_exposure ≈ total_component_vega_exposure
            target_index_contracts = total_component_vega_exposure / index_vega
            vega_ratio = total_component_vega_exposure / (index_vega * max(1, target_index_contracts))
            
            # Adjust index allocation proportionally
            adjusted_index_allocation = base_index_allocation * min(vega_ratio, 1.5)
            adjusted_index_allocation = min(adjusted_index_allocation, total_position * 0.6)  # Cap at 60%
        else:
            adjusted_index_allocation = base_index_allocation
            vega_ratio = 1.0
        
        # Calculate index position
        index_position = {
            'symbol': 'QQQ',
            'allocation': adjusted_index_allocation,
            'vega_per_contract': index_vega,
            'estimated_contracts': max(1, int(adjusted_index_allocation / 1000)),
            'vega_exposure': index_vega * max(1, int(adjusted_index_allocation / 1000)),
            'side': index_side
        }
        
        # Determine if vega-neutral
        is_vega_neutral = abs(vega_ratio - 1.0) < 0.2
        
        return {
            'signal_type': signal_type,
            'total_position': total_position,
            'index': index_position,
            'components': component_positions,
            'summary': {
                'is_vega_neutral': is_vega_neutral,
                'vega_ratio': vega_ratio,
                'index_allocation': adjusted_index_allocation,
                'component_allocation': base_component_allocation,
                'index_vega_exposure': index_position['vega_exposure'],
                'component_vega_exposure': total_component_vega_exposure,
                'num_components': len(component_positions)
            }
        }
    
    def _equal_weight_allocation(self, signal_type: str, total_position: float,
                                  index_side: str, component_side: str) -> Dict:
        """Fallback to equal-weight allocation when vega data unavailable."""
        index_allocation = total_position * 0.5
        component_allocation = total_position * 0.5
        per_component = component_allocation / self.num_components
        
        return {
            'signal_type': signal_type,
            'total_position': total_position,
            'index': {
                'symbol': 'QQQ',
                'allocation': index_allocation,
                'side': index_side,
                'vega_per_contract': 0,
                'estimated_contracts': 0,
                'vega_exposure': 0
            },
            'components': {
                f'COMP_{i}': {
                    'allocation': per_component,
                    'side': component_side,
                    'weight': 1 / self.num_components
                }
                for i in range(self.num_components)
            },
            'summary': {
                'is_vega_neutral': False,
                'vega_ratio': 1.0,
                'index_allocation': index_allocation,
                'component_allocation': component_allocation,
                'num_components': self.num_components,
                'note': 'Equal weighting (vega data unavailable)'
            }
        }
    
    def print_position_summary(self, positions: Dict):
        """Print a formatted summary of vega-weighted positions."""
        if not positions.get('can_trade', True):
            print(f"\n❌ Cannot trade: {positions.get('reason', 'Unknown')}")
            return
        
        print("\n" + "=" * 80)
        print("VEGA-WEIGHTED POSITION SIZING")
        print("=" * 80)
        
        # Kelly info
        kelly = positions.get('kelly_sizing', {})
        print(f"\n📊 Kelly Criterion:")
        print(f"   Kelly %: {kelly.get('kelly_pct', 0):.1%}")
        print(f"   Confidence Multiplier: {kelly.get('confidence_multiplier', 1):.2f}x")
        print(f"   Total Position: ${positions.get('total_position', 0):,.0f}")
        
        # Signal info
        print(f"\n📈 Signal: {positions.get('signal_type', 'N/A')}")
        
        # Index position
        index = positions.get('index', {})
        print(f"\n🏛️  INDEX POSITION:")
        print(f"   Symbol: {index.get('symbol', 'QQQ')}")
        print(f"   Side: {index.get('side', 'N/A')}")
        print(f"   Allocation: ${index.get('allocation', 0):,.0f}")
        print(f"   Est. Contracts: {index.get('estimated_contracts', 0)}")
        print(f"   Vega/Contract: {index.get('vega_per_contract', 0):.4f}")
        
        # Component positions
        components = positions.get('components', {})
        print(f"\n📦 COMPONENT POSITIONS ({len(components)} stocks):")
        print(f"   {'Symbol':<8} {'Side':<6} {'Allocation':>12} {'Weight':>8} {'Vega':>8}")
        print("   " + "-" * 50)
        
        # Show top 10 components
        sorted_components = sorted(
            components.items(),
            key=lambda x: x[1].get('allocation', 0),
            reverse=True
        )[:10]
        
        for symbol, data in sorted_components:
            alloc = data.get('allocation', 0)
            weight = data.get('weight', 0)
            vega = data.get('vega_per_contract', 0)
            side = data.get('side', 'N/A')
            print(f"   {symbol:<8} {side:<6} ${alloc:>10,.0f} {weight:>7.1%} {vega:>8.4f}")
        
        if len(components) > 10:
            print(f"   ... and {len(components) - 10} more")
        
        # Summary
        summary = positions.get('summary', {})
        print(f"\n📋 SUMMARY:")
        print(f"   Vega Neutral: {'✅ Yes' if summary.get('is_vega_neutral') else '❌ No'}")
        print(f"   Vega Ratio: {summary.get('vega_ratio', 0):.2f} (1.0 = perfectly neutral)")
        print(f"   Index Vega Exposure: {summary.get('index_vega_exposure', 0):.2f}")
        print(f"   Component Vega Exposure: {summary.get('component_vega_exposure', 0):.2f}")
        
        print("=" * 80)


# =============================================================================
# MULTI-STRATEGY VEGA-WEIGHTED SIZER
# =============================================================================

class MultiStrategyVegaSizer:
    """
    Calculates vega-weighted positions for all 3 strategies in parallel.
    """
    
    STRATEGIES = ['fixed', 'kelly_0.5x', 'kelly_1.0x']
    
    def __init__(self, base_capital: float = 100000):
        """Initialize sizers for each strategy."""
        self.base_capital = base_capital
        
        self.sizers = {
            'fixed': VegaWeightedKellySizer(
                base_capital=base_capital,
                kelly_fraction=0,  # Will use fixed 2%
                base_win_rate=0.669
            ),
            'kelly_0.5x': VegaWeightedKellySizer(
                base_capital=base_capital,
                kelly_fraction=0.5,
                base_win_rate=0.669
            ),
            'kelly_1.0x': VegaWeightedKellySizer(
                base_capital=base_capital,
                kelly_fraction=1.0,
                base_win_rate=0.669
            )
        }
    
    def calculate_all_positions(self,
                                 signal_type: str,
                                 current_capitals: Dict[str, float],
                                 ml_probability: float,
                                 use_ibkr: bool = False) -> Dict[str, Dict]:
        """
        Calculate vega-weighted positions for all strategies.
        
        Args:
            signal_type: 'SHORT_DISPERSION' or 'LONG_DISPERSION'
            current_capitals: Dict of strategy -> current capital
            ml_probability: ML model's probability
            use_ibkr: Whether to use live IBKR data
            
        Returns:
            Dict of strategy -> position sizing
        """
        results = {}
        
        for strategy in self.STRATEGIES:
            capital = current_capitals.get(strategy, self.base_capital)
            sizer = self.sizers[strategy]
            
            # For fixed strategy, override Kelly to use fixed 2%
            if strategy == 'fixed':
                # Temporarily set fixed position sizing
                original_kelly = sizer.risk_manager.kelly_fraction
                original_base = sizer.risk_manager.base_position_pct
                sizer.risk_manager.kelly_fraction = 0
                sizer.risk_manager.base_position_pct = 0.02
            
            positions = sizer.calculate_vega_weighted_positions(
                signal_type=signal_type,
                current_capital=capital,
                ml_probability=ml_probability,
                use_ibkr=use_ibkr
            )
            
            # Restore original settings for fixed strategy
            if strategy == 'fixed':
                sizer.risk_manager.kelly_fraction = original_kelly
                sizer.risk_manager.base_position_pct = original_base
            
            positions['strategy'] = strategy
            results[strategy] = positions
        
        return results
    
    def print_comparison(self, all_positions: Dict[str, Dict]):
        """Print comparison of all strategies."""
        print("\n" + "=" * 90)
        print("MULTI-STRATEGY VEGA-WEIGHTED COMPARISON")
        print("=" * 90)
        
        print(f"\n{'Strategy':<15} {'Total Pos':>12} {'Index':>12} {'Components':>12} {'Vega Neutral':>15}")
        print("-" * 70)
        
        for strategy in self.STRATEGIES:
            pos = all_positions.get(strategy, {})
            
            if not pos.get('can_trade', True):
                print(f"{strategy:<15} {'BLOCKED':>12} {'-':>12} {'-':>12} {'-':>15}")
                continue
            
            total = pos.get('total_position', 0)
            index_alloc = pos.get('index', {}).get('allocation', 0)
            comp_alloc = pos.get('summary', {}).get('component_allocation', 0)
            is_neutral = pos.get('summary', {}).get('is_vega_neutral', False)
            
            neutral_str = '✅ Yes' if is_neutral else '❌ No'
            
            print(f"{strategy:<15} ${total:>10,.0f} ${index_alloc:>10,.0f} ${comp_alloc:>10,.0f} {neutral_str:>15}")
        
        print("=" * 90)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("VEGA-WEIGHTED KELLY SIZER TEST")
    print("=" * 80)
    
    # Test single strategy
    sizer = VegaWeightedKellySizer(
        base_capital=100000,
        kelly_fraction=0.5,
        num_components=10
    )
    
    # Test with simulated data
    positions = sizer.calculate_vega_weighted_positions(
        signal_type='SHORT_DISPERSION',
        current_capital=100000,
        ml_probability=0.75,
        use_ibkr=False  # Use simulation
    )
    
    sizer.print_position_summary(positions)
    
    # Test multi-strategy
    print("\n\n")
    multi_sizer = MultiStrategyVegaSizer(base_capital=100000)
    
    all_positions = multi_sizer.calculate_all_positions(
        signal_type='SHORT_DISPERSION',
        current_capitals={'fixed': 100000, 'kelly_0.5x': 100000, 'kelly_1.0x': 100000},
        ml_probability=0.75,
        use_ibkr=False
    )
    
    multi_sizer.print_comparison(all_positions)
