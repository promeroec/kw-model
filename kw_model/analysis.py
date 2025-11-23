"""
Analysis tools for Kiyotaki-Wright model
Computes velocity, acceptability, liquidity, and other metrics
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class KWAnalyzer:
    """Analyzer for Kiyotaki-Wright economy metrics."""
    
    def __init__(self, economy):
        """
        Args:
            economy: KWEconomy instance
        """
        self.economy = economy
    
    def compute_stock(self, good: int) -> float:
        """
        Compute stock x_j = sum_i theta_ij of good j.
        
        Args:
            good: Good identifier
            
        Returns:
            Stock (proportion of population holding good j)
        """
        count = sum(1 for agent in self.economy.agents if agent.inventory == good)
        return count / len(self.economy.agents)
    
    def compute_offers(self, good: int) -> float:
        """
        Compute number of times good j gets offered in trade per period.
        
        Args:
            good: Good identifier
            
        Returns:
            Offer rate
        """
        # Count agents holding the good
        holders = [a for a in self.economy.agents if a.inventory == good]
        
        if not holders:
            return 0.0
        
        # In random bilateral matching, each agent is matched once per period
        # The offer happens if the holder wants to trade
        # We approximate this by looking at recent trading behavior
        
        offers = 0
        for holder in holders:
            # Simulate: what fraction of other agents would this holder want to trade with?
            for other_agent in self.economy.agents:
                if other_agent.agent_id == holder.agent_id:
                    continue
                
                if holder.wants_to_trade(good, other_agent.inventory):
                    offers += 1
        
        # Normalize by matching probability
        matching_prob = 1.0 / len(self.economy.agents)
        return offers * matching_prob
    
    def compute_transactions(self, good: int, window: int = 100) -> float:
        """
        Compute number of times good j gets traded per period.
        
        Args:
            good: Good identifier
            window: Number of recent periods to analyze
            
        Returns:
            Transaction rate
        """
        if self.economy.current_period < window:
            window = max(1, self.economy.current_period)
        
        transactions = 0
        
        # Look at trade history
        for agent in self.economy.agents:
            recent_trades = agent.trade_history[-window:] if len(agent.trade_history) > 0 else []
            
            for trade in recent_trades:
                if trade['action'] == 'trade':
                    if trade['gave'] == good or trade['received'] == good:
                        transactions += 1
        
        # Each trade is counted twice (once for each agent), so divide by 2
        return transactions / (2 * window)
    
    def compute_velocity(self, good: int, window: int = 100) -> float:
        """
        Compute velocity v_j = t_j / x_j.
        
        Args:
            good: Good identifier
            window: Number of periods for transaction calculation
            
        Returns:
            Velocity
        """
        stock = self.compute_stock(good)
        if stock == 0:
            return 0.0
        
        transactions = self.compute_transactions(good, window)
        return transactions / stock
    
    def compute_acceptability(self, good: int) -> float:
        """
        Compute acceptability a_j = t_j / o_j (probability of acceptance when offered).
        
        Args:
            good: Good identifier
            
        Returns:
            Acceptability (between 0 and 1)
        """
        offers = self.compute_offers(good)
        if offers == 0:
            return 0.0
        
        transactions = self.compute_transactions(good)
        return min(1.0, transactions / offers)
    
    def compute_liquidity(self, agent_type: int, good: int) -> float:
        """
        Compute liquidity d_ij: expected time for type i with good j 
        to trade for consumption good.
        
        Args:
            agent_type: Type of agent
            good: Starting good
            
        Returns:
            Expected duration (in periods)
        """
        # This requires solving a system of equations or simulation
        # We'll use a simplified Monte Carlo approach
        
        # Find consumption good for this type
        type_agents = [a for a in self.economy.agents if a.agent_type == agent_type]
        if not type_agents:
            return float('inf')
        
        consumption_good = type_agents[0].consumption_good
        
        # If already have consumption good, duration is 0
        if good == consumption_good:
            return 0.0
        
        # Monte Carlo simulation
        num_simulations = 1000
        durations = []
        
        for _ in range(num_simulations):
            current_good = good
            duration = 0
            max_duration = 100  # Prevent infinite loops
            
            while current_good != consumption_good and duration < max_duration:
                # Sample a random other agent
                other_agent = np.random.choice(self.economy.agents)
                other_good = other_agent.inventory
                
                # Create a temporary agent to check willingness
                temp_agent = type_agents[0]  # Use representative agent
                
                # Would temp_agent trade current_good for other_good?
                # And would other_agent trade?
                if (temp_agent.wants_to_trade(current_good, other_good) and
                    other_agent.wants_to_trade(other_good, current_good)):
                    current_good = other_good
                
                duration += 1
            
            durations.append(duration)
        
        return np.mean(durations)
    
    def compute_all_velocities(self) -> Dict[int, float]:
        """Compute velocities for all goods."""
        goods = [0, 1, 2, 3] if self.economy.fiat_money else [1, 2, 3]
        return {good: self.compute_velocity(good) for good in goods}
    
    def compute_all_acceptabilities(self) -> Dict[int, float]:
        """Compute acceptabilities for all goods."""
        goods = [0, 1, 2, 3] if self.economy.fiat_money else [1, 2, 3]
        return {good: self.compute_acceptability(good) for good in goods}
    
    def compute_all_stocks(self) -> Dict[int, float]:
        """Compute stocks for all goods."""
        goods = [0, 1, 2, 3] if self.economy.fiat_money else [1, 2, 3]
        return {good: self.compute_stock(good) for good in goods}
    
    def identify_media_of_exchange(self, acceptability_threshold: float = 0.5) -> List[int]:
        """
        Identify which goods serve as media of exchange.
        
        Args:
            acceptability_threshold: Minimum acceptability to be considered money
            
        Returns:
            List of goods serving as media of exchange
        """
        acceptabilities = self.compute_all_acceptabilities()
        
        media = []
        for good, accept in acceptabilities.items():
            if accept >= acceptability_threshold:
                media.append(good)
        
        return media
    
    def is_general_medium(self, good: int, threshold: float = 0.95) -> bool:
        """
        Check if good is a general medium of exchange (accepted by all).
        
        Args:
            good: Good identifier
            threshold: Minimum acceptability for general medium
            
        Returns:
            Boolean indicating if good is general medium
        """
        acceptability = self.compute_acceptability(good)
        return acceptability >= threshold
    
    def compute_equilibrium_type(self) -> str:
        """
        Determine the type of equilibrium based on inventory distribution.
        
        Returns:
            String describing equilibrium type
        """
        dist = self.economy.get_inventory_distribution()
        
        # Check for fiat money equilibrium
        if self.economy.fiat_money:
            fiat_stock = self.compute_stock(0)
            if fiat_stock > 0.1:  # Significant fiat money in circulation
                return "Fiat Money Equilibrium"
        
        # Check for commodity money patterns
        # Model A fundamental: p12=1, p23=0.5, p31=1
        if self.economy.model_type == 'A':
            p12 = dist.get((1, 2), 0)
            p13 = dist.get((1, 3), 0)
            p31 = dist.get((3, 1), 0)
            
            if p12 > 0.8 and p13 < 0.2 and p31 > 0.8:
                return "Model A Fundamental Equilibrium"
            elif p13 > 0.4:
                return "Model A Speculative Equilibrium"
        
        # Model B patterns
        elif self.economy.model_type == 'B':
            p13 = dist.get((1, 3), 0)
            p21 = dist.get((2, 1), 0)
            
            if p13 < 0.2 and p21 > 0.8:
                return "Model B Fundamental Equilibrium"
            elif p13 > 0.4:
                return "Model B Speculative Equilibrium"
        
        return "Transitional/Mixed Equilibrium"
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        report = {
            'equilibrium_type': self.compute_equilibrium_type(),
            'inventory_distribution': self.economy.get_inventory_distribution(),
            'stocks': self.compute_all_stocks(),
            'velocities': self.compute_all_velocities(),
            'acceptabilities': self.compute_all_acceptabilities(),
            'media_of_exchange': self.identify_media_of_exchange(),
            'general_media': [g for g in ([0, 1, 2, 3] if self.economy.fiat_money else [1, 2, 3])
                             if self.is_general_medium(g)],
            'welfare_by_type': self.economy.compute_welfare_by_type()
        }
        
        return report
    
    def print_report(self):
        """Print formatted analysis report."""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("KIYOTAKI-WRIGHT ECONOMY ANALYSIS")
        print("="*60)
        
        print(f"\nEquilibrium Type: {report['equilibrium_type']}")
        
        print("\nInventory Distribution (p_ij):")
        for agent_type in [1, 2, 3]:
            print(f"  Type {agent_type}:")
            goods = [0, 1, 2, 3] if self.economy.fiat_money else [1, 2, 3]
            for good in goods:
                prop = report['inventory_distribution'].get((agent_type, good), 0)
                print(f"    Good {good}: {prop:.4f}")
        
        print("\nStocks (x_j):")
        for good, stock in report['stocks'].items():
            print(f"  Good {good}: {stock:.4f}")
        
        print("\nVelocities (v_j):")
        for good, velocity in report['velocities'].items():
            print(f"  Good {good}: {velocity:.4f}")
        
        print("\nAcceptabilities (a_j):")
        for good, accept in report['acceptabilities'].items():
            print(f"  Good {good}: {accept:.4f}")
        
        print(f"\nMedia of Exchange: {report['media_of_exchange']}")
        print(f"General Media: {report['general_media']}")
        
        print("\nWelfare by Type:")
        for agent_type, welfare in report['welfare_by_type'].items():
            print(f"  Type {agent_type}: {welfare:.4f}")
        
        print("="*60 + "\n")
