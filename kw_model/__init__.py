"""
Kiyotaki-Wright Model: Agent-Based Implementation

An agent-based model of the seminal Kiyotaki and Wright (1989) 
"On Money as a Medium of Exchange" paper.

This package implements:
- Model A and Model B economies
- Fundamental and speculative equilibria
- Fiat money equilibria
- Velocity, acceptability, and liquidity calculations
- Welfare analysis

Example usage:
    ```python
    from kw_model import KWEconomy, KWAnalyzer
    from kw_model.visualization import plot_inventory_distribution
    
    # Create economy
    economy = KWEconomy(
        model_type='A',
        num_agents=300,
        strategy_name='fundamental'
    )
    
    # Run simulation
    economy.run_simulation(num_periods=1000, burn_in=100)
    
    # Analyze results
    analyzer = KWAnalyzer(economy)
    analyzer.print_report()
    
    # Visualize
    plot_inventory_distribution(economy)
    ```
"""

__version__ = "1.0.0"
__author__ = "Based on Kiyotaki & Wright (1989)"

from .agents import Agent
from .economy import KWEconomy
from .strategies import (
    TradingStrategy,
    FundamentalStrategy,
    SpeculativeStrategy,
    FiatMoneyStrategy,
    ValueFunctionStrategy,
    get_strategy
)
from .analysis import KWAnalyzer

__all__ = [
    'Agent',
    'KWEconomy',
    'KWAnalyzer',
    'TradingStrategy',
    'FundamentalStrategy',
    'SpeculativeStrategy',
    'FiatMoneyStrategy',
    'ValueFunctionStrategy',
    'get_strategy'
]
