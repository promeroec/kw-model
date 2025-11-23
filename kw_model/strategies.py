"""
Trading strategies for Kiyotaki-Wright model agents
"""
from typing import Dict, Optional
import numpy as np


class TradingStrategy:
    """Base class for trading strategies."""
    
    def __call__(self, agent, my_good: int, other_good: int, params: Optional[Dict] = None) -> bool:
        """
        Determine if agent should trade.
        
        Args:
            agent: The agent making the decision
            my_good: Good currently held
            other_good: Good offered by partner
            params: Additional parameters (e.g., beliefs about distribution)
            
        Returns:
            Boolean indicating willingness to trade
        """
        raise NotImplementedError


class FundamentalStrategy(TradingStrategy):
    """
    Fundamental strategy: Always prefer consumption good, 
    then prefer goods with lower storage costs.
    """
    
    def __call__(self, agent, my_good: int, other_good: int, params: Optional[Dict] = None) -> bool:
        # Always want consumption good
        if other_good == agent.consumption_good:
            return True
        
        # Never trade away consumption good
        if my_good == agent.consumption_good:
            return False
        
        # Don't trade for same good
        if my_good == other_good:
            return False
        
        # Prefer lower storage cost
        return agent.get_storage_cost(other_good) < agent.get_storage_cost(my_good)


class SpeculativeStrategy(TradingStrategy):
    """
    Speculative strategy for Model A: Type I agents trade good 2 for good 3
    even though good 3 has higher storage cost, because it's more marketable.
    
    For Model B: Types II and III speculate.
    """
    
    def __init__(self, model_type: str = 'A', agent_type: int = 1):
        """
        Args:
            model_type: 'A' or 'B'
            agent_type: Which type uses speculative strategy
        """
        self.model_type = model_type
        self.speculating_type = agent_type
    
    def __call__(self, agent, my_good: int, other_good: int, params: Optional[Dict] = None) -> bool:
        # Always want consumption good
        if other_good == agent.consumption_good:
            return True
        
        # Never trade away consumption good
        if my_good == agent.consumption_good:
            return False
        
        # Don't trade for same good
        if my_good == other_good:
            return False
        
        # Model A: Type I speculates by preferring good 3 over good 2
        if self.model_type == 'A' and agent.agent_type == 1:
            if other_good == 3:  # Good 3 is more marketable
                return True
            elif my_good == 3:  # Don't give up good 3
                return False
            # Otherwise use fundamental
            return agent.get_storage_cost(other_good) < agent.get_storage_cost(my_good)
        
        # Model B: Types II and III speculate
        elif self.model_type == 'B':
            if agent.agent_type == 2:
                # Type II prefers good 3 over good 1
                if other_good == 3:
                    return True
                elif my_good == 3:
                    return False
            elif agent.agent_type == 3:
                # Type III prefers good 2 over good 1
                if other_good == 2:
                    return True
                elif my_good == 2:
                    return False
            
            # Otherwise fundamental
            return agent.get_storage_cost(other_good) < agent.get_storage_cost(my_good)
        
        # Default to fundamental
        return agent.get_storage_cost(other_good) < agent.get_storage_cost(my_good)


class FiatMoneyStrategy(TradingStrategy):
    """
    Strategy when fiat money (good 0) is present.
    Assumes good 0 has lowest storage cost and highest acceptability.
    """
    
    def __call__(self, agent, my_good: int, other_good: int, params: Optional[Dict] = None) -> bool:
        # Always want consumption good most
        if other_good == agent.consumption_good:
            return True
        
        # Never trade away consumption good
        if my_good == agent.consumption_good:
            return False
        
        # Don't trade for same good
        if my_good == other_good:
            return False
        
        # Fiat money (good 0) is second-best after consumption good
        if other_good == 0:
            return True
        
        if my_good == 0:
            return False
        
        # Among real goods, prefer lower storage cost
        return agent.get_storage_cost(other_good) < agent.get_storage_cost(my_good)


class ValueFunctionStrategy(TradingStrategy):
    """
    Strategy based on computed value functions V_ij.
    This requires knowledge of the steady-state distribution.
    """
    
    def __init__(self, value_functions: Optional[Dict] = None):
        """
        Args:
            value_functions: Dict mapping (agent_type, good) -> value
        """
        self.value_functions = value_functions or {}
    
    def update_value_functions(self, value_functions: Dict):
        """Update the value function estimates."""
        self.value_functions = value_functions
    
    def __call__(self, agent, my_good: int, other_good: int, params: Optional[Dict] = None) -> bool:
        # If we don't have value functions yet, use fundamental
        if not self.value_functions:
            return FundamentalStrategy()(agent, my_good, other_good, params)
        
        # Don't trade for same good
        if my_good == other_good:
            return False
        
        # Get value functions
        v_my = self.value_functions.get((agent.agent_type, my_good), 0)
        v_other = self.value_functions.get((agent.agent_type, other_good), 0)
        
        return v_other > v_my


def get_strategy(strategy_name: str, **kwargs) -> TradingStrategy:
    """
    Factory function to get strategy by name.
    
    Args:
        strategy_name: Name of strategy ('fundamental', 'speculative', 'fiat', 'value')
        **kwargs: Additional arguments for strategy
        
    Returns:
        TradingStrategy instance
    """
    strategies = {
        'fundamental': FundamentalStrategy,
        'speculative': SpeculativeStrategy,
        'fiat': FiatMoneyStrategy,
        'value': ValueFunctionStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Choose from {list(strategies.keys())}")
    
    return strategies[strategy_name](**kwargs)
