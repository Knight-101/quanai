import numpy as np
from scipy.stats import norm
from typing import Dict

class LiquidationRiskAnalyzer:
    def __init__(self, maintenance_margin: float = 0.05):
        self.maintenance_margin = maintenance_margin
        
    def calculate_liquidation_price(self, position: Dict) -> float:
        """Calculate liquidation price for a position"""
        entry_price = position['entry_price']
        leverage = position['leverage']
        side = position['side']
        
        margin_ratio = 1/leverage
        liquidation_pct = margin_ratio + self.maintenance_margin
        
        if side == 'long':
            return entry_price * (1 - liquidation_pct)
        else:
            return entry_price * (1 + liquidation_pct)
        
    def portfolio_liquidation_risk(self, positions: Dict[str, Dict], 
                                  volatility: Dict[str, float]) -> float:
        """Calculate probability of liquidation within next time period"""
        risks = []
        for asset, pos in positions.items():
            liq_price = self.calculate_liquidation_price(pos)
            current_price = pos['mark_price']
            vol = volatility[asset]
            
            if pos['side'] == 'long':
                distance = (current_price - liq_price) / current_price
            else:
                distance = (liq_price - current_price) / current_price
                
            # Probability of crossing liquidation price
            prob = norm.cdf(-distance / vol)
            risks.append(prob)
            
        return np.max(risks)  # Most vulnerable position
    
    def stress_test_liquidations(self, positions: Dict[str, Dict],
                                price_shocks: Dict[str, float]) -> Dict[str, bool]:
        """Test if positions would liquidate under shock scenarios"""
        results = {}
        for asset, pos in positions.items():
            shocked_price = pos['mark_price'] * (1 + price_shocks[asset])
            liq_price = self.calculate_liquidation_price(pos)
            
            if pos['side'] == 'long' and shocked_price <= liq_price:
                results[asset] = True
            elif pos['side'] == 'short' and shocked_price >= liq_price:
                results[asset] = True
            else:
                results[asset] = False
                
        return results