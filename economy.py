"""
Economy class for Kiyotaki-Wright model
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from .agents import Agent
from .strategies import get_strategy


class KWEconomy:
    """
    Kiyotaki-Wright Economy with random bilateral matching.
    
    Attributes:
        model_type: 'A' or 'B' (determines production specialties)
        num_agents: Total number of agents
        beta: Discount factor
        agents: List of Agent objects
        current_period: Current time period
    """
    
    def __init__(
        self,
        model_type: str = 'A',
        num_agents: int = 300,
        storage_costs: Optional[Dict[Tuple[int, int], float]] = None,
        utilities: Optional[Dict[int, float]] = None,
        disutilities: Optional[Dict[int, float]] = None,
        beta: float = 0.95,
        strategy_name: str = 'fundamental',
        fiat_money: bool = False,
        fiat_proportion: float = 0.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the economy.
        
        Args:
            model_type: 'A' or 'B'
            num_agents: Number of agents (should be divisible by 3)
            storage_costs: Dict mapping (agent_type, good) -> storage cost
            utilities: Dict mapping agent_type -> utility from consumption
            disutilities: Dict mapping agent_type -> disutility from production
            beta: Discount factor
            strategy_name: Trading strategy ('fundamental', 'speculative', 'fiat')
            fiat_money: Whether to include fiat money (good 0)
            fiat_proportion: Proportion of agents initially holding fiat money
            random_seed: Random seed for reproducibility
        """
        if num_agents % 3 != 0:
            raise ValueError("num_agents must be divisible by 3")
        
        self.model_type = model_type.upper()
        if self.model_type not in ['A', 'B']:
            raise ValueError("model_type must be 'A' or 'B'")
        
        self.num_agents = num_agents
        self.beta = beta
        self.fiat_money = fiat_money
        self.fiat_proportion = fiat_proportion
        self.current_period = 0
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Set default parameters if not provided
        self.storage_costs = storage_costs or self._default_storage_costs()
        self.utilities = utilities or {1: 10.0, 2: 10.0, 3: 10.0}
        self.disutilities = disutilities or {1: 2.0, 2: 2.0, 3: 2.0}
        
        # Initialize agents
        self.agents = self._initialize_agents(strategy_name)
        
        # Statistics
        self.history = defaultdict(list)
        
    def _default_storage_costs(self) -> Dict[Tuple[int, int], float]:
        """
        Set up default storage costs as in the paper.
        c_i3 > c_i2 > c_i1 > 0 for all i
        """
        costs = {}
        
        # Fiat money has zero storage cost
        if self.fiat_money:
            for i in [1, 2, 3]:
                costs[(i, 0)] = 0.0
        
        # Real goods
        for i in [1, 2, 3]:
            costs[(i, 1)] = 0.5   # Most storable real good
            costs[(i, 2)] = 1.0   # Medium storability
            costs[(i, 3)] = 2.0   # Least storable real good
        
        return costs
    
    def _get_production_consumption_mapping(self) -> Dict[int, Dict[str, int]]:
        """Get production and consumption mappings based on model type."""
        if self.model_type == 'A':
            # Model A: Type 1 produces 2, Type 2 produces 3, Type 3 produces 1
            return {
                1: {'produces': 2, 'consumes': 1},
                2: {'produces': 3, 'consumes': 2},
                3: {'produces': 1, 'consumes': 3}
            }
        else:  # Model B
            # Model B: Type 1 produces 3, Type 2 produces 1, Type 3 produces 2
            return {
                1: {'produces': 3, 'consumes': 1},
                2: {'produces': 1, 'consumes': 2},
                3: {'produces': 2, 'consumes': 3}
            }
    
    def _initialize_agents(self, strategy_name: str) -> List[Agent]:
        """Initialize all agents."""
        agents = []
        mapping = self._get_production_consumption_mapping()
        agents_per_type = self.num_agents // 3
        
        # Determine how many agents start with fiat money
        num_fiat_holders = int(self.num_agents * self.fiat_proportion) if self.fiat_money else 0
        
        agent_id = 0
        for agent_type in [1, 2, 3]:
            prod_good = mapping[agent_type]['produces']
            cons_good = mapping[agent_type]['consumes']
            
            # Create storage cost dict for this agent type
            agent_storage_costs = {
                good: self.storage_costs.get((agent_type, good), 0.0)
                for good in ([0, 1, 2, 3] if self.fiat_money else [1, 2, 3])
            }
            
            for _ in range(agents_per_type):
                # Determine initial inventory
                if agent_id < num_fiat_holders:
                    initial_inventory = 0  # Fiat money
                else:
                    initial_inventory = prod_good  # Start with production good
                
                # Get strategy
                if strategy_name == 'speculative':
                    strategy = get_strategy('speculative', 
                                          model_type=self.model_type,
                                          agent_type=agent_type)
                elif strategy_name == 'fiat':
                    strategy = get_strategy('fiat')
                else:
                    strategy = get_strategy('fundamental')
                
                agent = Agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    production_good=prod_good,
                    consumption_good=cons_good,
                    storage_costs=agent_storage_costs,
                    utility=self.utilities[agent_type],
                    disutility=self.disutilities[agent_type],
                    initial_inventory=initial_inventory,
                    strategy=strategy
                )
                
                agents.append(agent)
                agent_id += 1
        
        return agents
    
    def run_period(self) -> Dict:
        """
        Run one period of the simulation.
        
        Returns:
            Dictionary of statistics for this period
        """
        # Step 1: Random bilateral matching and trading
        trades = self._matching_and_trading()
        
        # Step 2: Consumption and production
        consumption_utility = 0.0
        production_disutility = 0.0
        for agent in self.agents:
            u, d = agent.consume_and_produce()
            consumption_utility += u
            production_disutility += d
        
        # Step 3: Accrue storage costs
        storage_costs = sum(agent.accrue_storage_cost() for agent in self.agents)
        
        # Step 4: Update inventory history
        for agent in self.agents:
            agent.inventory_history.append(agent.inventory)
        
        # Collect statistics
        stats = {
            'period': self.current_period,
            'num_trades': trades,
            'consumption_utility': consumption_utility,
            'production_disutility': production_disutility,
            'storage_costs': storage_costs,
            'net_utility': consumption_utility - production_disutility - storage_costs,
            'inventory_distribution': self.get_inventory_distribution()
        }
        
        self.current_period += 1
        
        return stats
    
    def _matching_and_trading(self) -> int:
        """
        Randomly match agents in pairs and execute trades.
        
        Returns:
            Number of successful trades
        """
        # Shuffle agents for random matching
        shuffled_agents = np.random.permutation(self.agents)
        
        # Pair up agents
        num_trades = 0
        for i in range(0, len(shuffled_agents) - 1, 2):
            agent1 = shuffled_agents[i]
            agent2 = shuffled_agents[i + 1]
            
            # Attempt trade
            if agent1.trade(agent2):
                num_trades += 1
        
        return num_trades
    
    def run_simulation(self, num_periods: int, burn_in: int = 100) -> Dict:
        """
        Run the simulation for multiple periods.
        
        Args:
            num_periods: Total number of periods to simulate
            burn_in: Number of initial periods to discard
            
        Returns:
            Dictionary of aggregated statistics
        """
        all_stats = []
        
        for _ in range(num_periods):
            stats = self.run_period()
            
            # Only record after burn-in
            if self.current_period > burn_in:
                all_stats.append(stats)
                
                # Record in history
                for key, value in stats.items():
                    if key != 'inventory_distribution':
                        self.history[key].append(value)
        
        return self._aggregate_statistics(all_stats)
    
    def _aggregate_statistics(self, stats_list: List[Dict]) -> Dict:
        """Aggregate statistics across periods."""
        if not stats_list:
            return {}
        
        aggregated = {
            'avg_trades_per_period': np.mean([s['num_trades'] for s in stats_list]),
            'avg_consumption_utility': np.mean([s['consumption_utility'] for s in stats_list]),
            'avg_storage_costs': np.mean([s['storage_costs'] for s in stats_list]),
            'avg_net_utility': np.mean([s['net_utility'] for s in stats_list]),
            'steady_state_distribution': self.get_inventory_distribution(),
            'welfare_by_type': self.compute_welfare_by_type()
        }
        
        return aggregated
    
    def get_inventory_distribution(self) -> Dict[Tuple[int, int], float]:
        """
        Get the current inventory distribution p_ij.
        
        Returns:
            Dictionary mapping (agent_type, good) -> proportion
        """
        distribution = defaultdict(int)
        counts_by_type = defaultdict(int)
        
        for agent in self.agents:
            distribution[(agent.agent_type, agent.inventory)] += 1
            counts_by_type[agent.agent_type] += 1
        
        # Convert to proportions
        proportions = {
            key: count / counts_by_type[key[0]]
            for key, count in distribution.items()
        }
        
        return dict(proportions)
    
    def compute_welfare_by_type(self) -> Dict[int, float]:
        """
        Compute steady-state welfare W_i for each type.
        
        Returns:
            Dictionary mapping agent_type -> welfare
        """
        welfare = {}
        
        for agent_type in [1, 2, 3]:
            type_agents = [a for a in self.agents if a.agent_type == agent_type]
            
            if not type_agents:
                welfare[agent_type] = 0.0
                continue
            
            # Average per-period utility
            avg_consumption = np.mean([a.consumption_count for a in type_agents])
            avg_production = np.mean([a.production_count for a in type_agents])
            avg_storage = np.mean([a.total_storage_cost for a in type_agents])
            
            # Normalize by number of periods
            periods = max(1, self.current_period)
            
            utility = (avg_consumption / periods) * self.utilities[agent_type]
            disutility = (avg_production / periods) * self.disutilities[agent_type]
            storage = avg_storage / periods
            
            # Steady-state welfare (simplified)
            welfare[agent_type] = (utility - disutility - storage) / (1 - self.beta)
        
        return welfare
    
    def reset(self):
        """Reset the economy for a new simulation."""
        self.current_period = 0
        self.history = defaultdict(list)
        
        for agent in self.agents:
            agent.reset_statistics()
            # Reset inventory to production good or fiat money
            if self.fiat_money and np.random.random() < self.fiat_proportion:
                agent.inventory = 0
            else:
                agent.inventory = agent.production_good
    
    def get_summary(self) -> str:
        """Get a summary of the economy."""
        dist = self.get_inventory_distribution()
        
        summary = f"Kiyotaki-Wright Economy (Model {self.model_type})\n"
        summary += f"{'='*50}\n"
        summary += f"Number of agents: {self.num_agents}\n"
        summary += f"Current period: {self.current_period}\n"
        summary += f"Fiat money: {self.fiat_money}\n\n"
        
        summary += "Inventory Distribution:\n"
        for agent_type in [1, 2, 3]:
            summary += f"  Type {agent_type}:\n"
            for good in ([0, 1, 2, 3] if self.fiat_money else [1, 2, 3]):
                prop = dist.get((agent_type, good), 0.0)
                summary += f"    Good {good}: {prop:.3f}\n"
        
        return summary
