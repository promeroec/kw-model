"""
Visualization tools for Kiyotaki-Wright model
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from matplotlib.patches import Rectangle


def plot_inventory_distribution(
    economy,
    title: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot the inventory distribution p_ij.
    
    Args:
        economy: KWEconomy instance
        title: Plot title
        figsize: Figure size
    """
    dist = economy.get_inventory_distribution()
    
    # Prepare data
    agent_types = [1, 2, 3]
    goods = [0, 1, 2, 3] if economy.fiat_money else [1, 2, 3]
    
    data = np.zeros((len(agent_types), len(goods)))
    for i, agent_type in enumerate(agent_types):
        for j, good in enumerate(goods):
            data[i, j] = dist.get((agent_type, good), 0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(len(goods)))
    ax.set_yticks(range(len(agent_types)))
    ax.set_xticklabels([f'Good {g}' for g in goods])
    ax.set_yticklabels([f'Type {t}' for t in agent_types])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(agent_types)):
        for j in range(len(goods)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_xlabel('Goods')
    ax.set_ylabel('Agent Types')
    
    if title is None:
        title = f'Inventory Distribution (Model {economy.model_type})'
    ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax


def plot_time_series(
    economy,
    metrics: List[str] = ['num_trades', 'net_utility'],
    figsize: tuple = (12, 6)
):
    """
    Plot time series of various metrics.
    
    Args:
        economy: KWEconomy instance
        metrics: List of metrics to plot
        figsize: Figure size
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in economy.history:
            data = economy.history[metric]
            axes[i].plot(data, linewidth=2)
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Period')
    fig.suptitle(f'Time Series (Model {economy.model_type})')
    plt.tight_layout()
    return fig, axes


def plot_velocity_vs_real_balances(
    results: Dict[float, Dict],
    figsize: tuple = (10, 6)
):
    """
    Plot velocity as a function of real balances (recreate Figure 9c from paper).
    
    Args:
        results: Dictionary mapping real_balance -> metrics
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    real_balances = sorted(results.keys())
    
    # Extract velocities for each good
    goods = [0, 1, 2, 3]
    for good in goods:
        velocities = [results[s]['velocities'].get(good, 0) for s in real_balances]
        label = 'Fiat Money' if good == 0 else f'Good {good}'
        ax.plot(real_balances, velocities, marker='o', linewidth=2, label=label)
    
    ax.set_xlabel('Real Balances (S = M/P)')
    ax.set_ylabel('Velocity')
    ax.set_title('Velocity vs Real Balances')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_acceptability_vs_real_balances(
    results: Dict[float, Dict],
    figsize: tuple = (10, 6)
):
    """
    Plot acceptability as a function of real balances (recreate Figure 9d from paper).
    
    Args:
        results: Dictionary mapping real_balance -> metrics
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    real_balances = sorted(results.keys())
    
    # Extract acceptabilities for each good
    goods = [0, 1, 2, 3]
    for good in goods:
        acceptabilities = [results[s]['acceptabilities'].get(good, 0) for s in real_balances]
        label = 'Fiat Money' if good == 0 else f'Good {good}'
        ax.plot(real_balances, acceptabilities, marker='o', linewidth=2, label=label)
    
    ax.set_xlabel('Real Balances (S = M/P)')
    ax.set_ylabel('Acceptability')
    ax.set_title('Acceptability vs Real Balances')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig, ax


def plot_stocks_vs_real_balances(
    results: Dict[float, Dict],
    figsize: tuple = (10, 6)
):
    """
    Plot stocks as a function of real balances (recreate Figure 9a from paper).
    
    Args:
        results: Dictionary mapping real_balance -> metrics
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    real_balances = sorted(results.keys())
    
    # Extract stocks for each good
    goods = [0, 1, 2, 3]
    for good in goods:
        stocks = [results[s]['stocks'].get(good, 0) for s in real_balances]
        label = 'Fiat Money' if good == 0 else f'Good {good}'
        ax.plot(real_balances, stocks, marker='o', linewidth=2, label=label)
    
    ax.set_xlabel('Real Balances (S = M/P)')
    ax.set_ylabel('Stock (x_j)')
    ax.set_title('Stocks vs Real Balances')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_welfare_comparison(
    welfare_dict: Dict[str, Dict[int, float]],
    figsize: tuple = (10, 6)
):
    """
    Compare welfare across different equilibria.
    
    Args:
        welfare_dict: Dictionary mapping equilibrium_name -> {agent_type -> welfare}
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    equilibria = list(welfare_dict.keys())
    agent_types = [1, 2, 3]
    
    x = np.arange(len(agent_types))
    width = 0.8 / len(equilibria)
    
    for i, (eq_name, welfare) in enumerate(welfare_dict.items()):
        values = [welfare.get(t, 0) for t in agent_types]
        offset = width * (i - len(equilibria)/2 + 0.5)
        ax.bar(x + offset, values, width, label=eq_name)
    
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Welfare')
    ax.set_title('Welfare Comparison Across Equilibria')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Type {t}' for t in agent_types])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, ax


def plot_trading_matrix(
    economy,
    figsize: tuple = (12, 4)
):
    """
    Visualize the trading matrices as in Figure 1 of the paper.
    
    Args:
        economy: KWEconomy instance
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Three possible meetings: (I, II), (II, III), (III, I)
    meetings = [
        (1, 2, "Type I meets Type II"),
        (2, 3, "Type II meets Type III"),
        (3, 1, "Type III meets Type I")
    ]
    
    goods = [1, 2, 3]
    
    for idx, (type1, type2, title) in enumerate(meetings):
        ax = axes[idx]
        
        # Get representative agents
        agent1 = next(a for a in economy.agents if a.agent_type == type1)
        agent2 = next(a for a in economy.agents if a.agent_type == type2)
        
        # Create matrix: rows are type1's goods, columns are type2's goods
        type1_goods = [g for g in goods if g != agent1.consumption_good]
        type2_goods = [g for g in goods if g != agent2.consumption_good]
        
        matrix = np.zeros((len(type1_goods), len(type2_goods)))
        labels = [['' for _ in range(len(type2_goods))] for _ in range(len(type1_goods))]
        
        for i, good1 in enumerate(type1_goods):
            for j, good2 in enumerate(type2_goods):
                # Check if both want to trade
                wants1 = agent1.wants_to_trade(good1, good2)
                wants2 = agent2.wants_to_trade(good2, good1)
                
                if wants1 and wants2:
                    matrix[i, j] = 1
                    labels[i][j] = 'T'
                else:
                    matrix[i, j] = 0
                    labels[i][j] = 'N'
                
                # Check for double coincidence
                if good1 == agent2.consumption_good and good2 == agent1.consumption_good:
                    matrix[i, j] = 2
                    labels[i][j] = 'DC'
        
        # Plot
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=2, aspect='auto')
        
        # Set ticks
        ax.set_xticks(range(len(type2_goods)))
        ax.set_yticks(range(len(type1_goods)))
        ax.set_xticklabels([f'G{g}' for g in type2_goods])
        ax.set_yticklabels([f'G{g}' for g in type1_goods])
        
        # Add text
        for i in range(len(type1_goods)):
            for j in range(len(type2_goods)):
                ax.text(j, i, labels[i][j], ha="center", va="center",
                       color="black", fontsize=12, fontweight='bold')
        
        ax.set_xlabel(f'Type {type2} has')
        ax.set_ylabel(f'Type {type1} has')
        ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    return fig, axes


def plot_convergence(
    economy,
    metric: str = 'inventory_distribution',
    figsize: tuple = (12, 6)
):
    """
    Plot convergence to steady state.
    
    Args:
        economy: KWEconomy instance
        metric: Metric to track convergence
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if metric == 'inventory_distribution':
        # Track how p_ij evolves over time
        # This requires storing history, which we'll add
        ax.text(0.5, 0.5, 'Feature in development', 
               transform=ax.transAxes, ha='center', va='center')
    
    ax.set_xlabel('Period')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Convergence to Steady State')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def create_figure_9_style(results: Dict[float, Dict], figsize: tuple = (12, 10)):
    """
    Create a comprehensive figure similar to Figure 9 in the paper.
    
    Args:
        results: Dictionary mapping real_balance -> metrics
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot a: Stocks
    ax1 = fig.add_subplot(gs[0, 0])
    real_balances = sorted(results.keys())
    goods = [0, 1, 2, 3]
    for good in goods:
        stocks = [results[s]['stocks'].get(good, 0) for s in real_balances]
        label = 'Fiat Money' if good == 0 else f'Good {good}'
        ax1.plot(real_balances, stocks, marker='o', linewidth=2, label=label)
    ax1.set_xlabel('Real Balances (S)')
    ax1.set_ylabel('Stock (x_j)')
    ax1.set_title('a. Stocks')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot b: Transactions (approximate)
    ax2 = fig.add_subplot(gs[0, 1])
    for good in goods:
        trans = [results[s].get('transactions', {}).get(good, 0) for s in real_balances]
        label = 'Fiat Money' if good == 0 else f'Good {good}'
        ax2.plot(real_balances, trans, marker='o', linewidth=2, label=label)
    ax2.set_xlabel('Real Balances (S)')
    ax2.set_ylabel('Transactions (t_j)')
    ax2.set_title('b. Transactions')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot c: Velocities
    ax3 = fig.add_subplot(gs[1, 0])
    for good in goods:
        velocities = [results[s]['velocities'].get(good, 0) for s in real_balances]
        label = 'Fiat Money' if good == 0 else f'Good {good}'
        ax3.plot(real_balances, velocities, marker='o', linewidth=2, label=label)
    ax3.set_xlabel('Real Balances (S)')
    ax3.set_ylabel('Velocity (v_j)')
    ax3.set_title('c. Velocities')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot d: Acceptabilities
    ax4 = fig.add_subplot(gs[1, 1])
    for good in goods:
        acceptabilities = [results[s]['acceptabilities'].get(good, 0) for s in real_balances]
        label = 'Fiat Money' if good == 0 else f'Good {good}'
        ax4.plot(real_balances, acceptabilities, marker='o', linewidth=2, label=label)
    ax4.set_xlabel('Real Balances (S)')
    ax4.set_ylabel('Acceptability (a_j)')
    ax4.set_title('d. Acceptabilities')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    plt.suptitle('Monetary Characteristics vs Real Balances', fontsize=14, y=0.995)
    
    return fig


def save_all_plots(economy, analyzer, prefix: str = 'kw_'):
    """
    Generate and save all standard plots.
    
    Args:
        economy: KWEconomy instance
        analyzer: KWAnalyzer instance
        prefix: Filename prefix
    """
    plots = []
    
    # Inventory distribution
    fig, _ = plot_inventory_distribution(economy)
    fig.savefig(f'{prefix}inventory_dist.png', dpi=150, bbox_inches='tight')
    plots.append(f'{prefix}inventory_dist.png')
    plt.close(fig)
    
    # Time series
    if economy.history:
        fig, _ = plot_time_series(economy)
        fig.savefig(f'{prefix}time_series.png', dpi=150, bbox_inches='tight')
        plots.append(f'{prefix}time_series.png')
        plt.close(fig)
    
    # Trading matrix
    fig, _ = plot_trading_matrix(economy)
    fig.savefig(f'{prefix}trading_matrix.png', dpi=150, bbox_inches='tight')
    plots.append(f'{prefix}trading_matrix.png')
    plt.close(fig)
    
    return plots
