"""
Agent class for Kiyotaki-Wright model
"""
import numpy as np
from typing import Optional, Dict, Tuple


class Agent:
    """
    Represents an agent in the Kiyotaki-Wright economy.
    
    Attributes:
        agent_id: Unique identifier
        agent_type: Type of agent (1, 2, or 3)
        consumption_good: Good that provides utility
        production_good: Good that agent produces
        inventory: Currently held good
        storage_costs: Dictionary of storage costs for each good
        utility: Utility from consuming consumption good
        disutility: Disutility from producing
        strategy: Trading strategy function
    """
    
    def __init__(
        self,
        agent_id: int,
        agent_type: int,
        production_good: int,
        consumption_good: int,
        storage_costs: Dict[int, float],
        utility: float,
        disutility: float,
        initial_inventory: int,
        strategy: Optional[callable] = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.consumption_good = consumption_good
        self.production_good = production_good
        self.inventory = initial_inventory
        self.storage_costs = storage_costs
        self.utility = utility
        self.disutility = disutility
        self.strategy = strategy
        
        # Statistics tracking
        self.consumption_count = 0
        self.production_count = 0
        self.total_storage_cost = 0.0
        self.trade_history = []
        self.inventory_history = [initial_inventory]
        
    def get_storage_cost(self, good: int) -> float:
        """Get storage cost for a specific good."""
        return self.storage_costs.get(good, 0.0)
    
    def current_storage_cost(self) -> float:
        """Get current storage cost based on inventory."""
        return self.get_storage_cost(self.inventory)
    
    def wants_to_trade(
        self,
        my_good: int,
        other_good: int,
        strategy_params: Optional[Dict] = None
    ) -> bool:
        """
        Determine if agent wants to trade my_good for other_good.
        
        Args:
            my_good: Good currently held
            other_good: Good offered by trading partner
            strategy_params: Additional parameters for strategy
            
        Returns:
            Boolean indicating willingness to trade
        """
        if self.strategy is None:
            # Default: fundamental strategy
            return self._fundamental_strategy(my_good, other_good)
        else:
            return self.strategy(self, my_good, other_good, strategy_params)
    
    def _fundamental_strategy(self, my_good: int, other_good: int) -> bool:
        """
        Fundamental strategy: prefer consumption good, then lower storage cost.
        """
        # Always want consumption good
        if other_good == self.consumption_good:
            return True
        
        # Never trade consumption good for anything else
        if my_good == self.consumption_good:
            return False
        
        # Never trade for same good
        if my_good == other_good:
            return False
        
        # Prefer lower storage cost
        return self.get_storage_cost(other_good) < self.get_storage_cost(my_good)
    
    def consume_and_produce(self) -> Tuple[float, float]:
        """
        Consume current good if it's consumption good, then produce.
        
        Returns:
            Tuple of (utility gained, disutility incurred)
        """
        utility_gained = 0.0
        disutility_incurred = 0.0
        
        if self.inventory == self.consumption_good:
            # Consume
            utility_gained = self.utility
            self.consumption_count += 1
            
            # Produce
            disutility_incurred = self.disutility
            self.production_count += 1
            self.inventory = self.production_good
            
            self.trade_history.append({
                'action': 'consume_produce',
                'utility': utility_gained - disutility_incurred
            })
        
        return utility_gained, disutility_incurred
    
    def trade(self, other_agent: 'Agent') -> bool:
        """
        Attempt to trade with another agent.
        
        Returns:
            Boolean indicating if trade occurred
        """
        my_good = self.inventory
        other_good = other_agent.inventory
        
        # Check if both want to trade
        i_want = self.wants_to_trade(my_good, other_good)
        other_wants = other_agent.wants_to_trade(other_good, my_good)
        
        if i_want and other_wants:
            # Execute trade
            self.inventory = other_good
            other_agent.inventory = my_good
            
            # Record trade
            self.trade_history.append({
                'action': 'trade',
                'partner': other_agent.agent_id,
                'gave': my_good,
                'received': other_good
            })
            other_agent.trade_history.append({
                'action': 'trade',
                'partner': self.agent_id,
                'gave': other_good,
                'received': my_good
            })
            
            return True
        
        return False
    
    def accrue_storage_cost(self) -> float:
        """Accrue storage cost for current inventory."""
        cost = self.current_storage_cost()
        self.total_storage_cost += cost
        return cost
    
    def get_statistics(self) -> Dict:
        """Get agent statistics."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'consumption_count': self.consumption_count,
            'production_count': self.production_count,
            'total_storage_cost': self.total_storage_cost,
            'current_inventory': self.inventory,
            'num_trades': len([t for t in self.trade_history if t['action'] == 'trade'])
        }
    
    def reset_statistics(self):
        """Reset statistics for new simulation."""
        self.consumption_count = 0
        self.production_count = 0
        self.total_storage_cost = 0.0
        self.trade_history = []
        self.inventory_history = [self.inventory]
    
    def __repr__(self):
        return (f"Agent(id={self.agent_id}, type={self.agent_type}, "
                f"inventory={self.inventory}, consumes={self.consumption_good})")
