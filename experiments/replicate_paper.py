"""
Replicate main results from Kiyotaki & Wright (1989)

This script replicates:
- Theorem 1: Model A fundamental and speculative equilibria
- Theorem 2: Model B fundamental and speculative equilibria  
- Theorem 3: Fiat money equilibrium
- Figure 9: Velocity, acceptability vs real balances
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from kw_model import KWEconomy, KWAnalyzer
from kw_model.visualization import (
    plot_inventory_distribution,
    create_figure_9_style,
    plot_welfare_comparison,
    plot_trading_matrix
)


def experiment_1_model_a_fundamental():
    """
    Replicate Model A fundamental equilibrium (Theorem 1a).
    Should have: p12=1, p23=0.5, p31=1
    Good 1 serves as unique commodity money.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Model A Fundamental Equilibrium")
    print("="*70)
    
    # Set parameters to ensure fundamental equilibrium
    # c13 - c12 > 0.5*beta*u1
    storage_costs = {
        (1, 1): 0.5, (1, 2): 1.0, (1, 3): 3.0,  # Large difference
        (2, 1): 0.5, (2, 2): 1.0, (2, 3): 3.0,
        (3, 1): 0.5, (3, 2): 1.0, (3, 3): 3.0
    }
    
    economy = KWEconomy(
        model_type='A',
        num_agents=300,
        storage_costs=storage_costs,
        utilities={1: 10.0, 2: 10.0, 3: 10.0},
        beta=0.95,
        strategy_name='fundamental',
        random_seed=42
    )
    
    # Run simulation
    results = economy.run_simulation(num_periods=2000, burn_in=500)
    
    # Analyze
    analyzer = KWAnalyzer(economy)
    report = analyzer.generate_report()
    
    print(f"\nEquilibrium type: {report['equilibrium_type']}")
    print(f"Media of exchange: {report['media_of_exchange']}")
    
    # Check if matches theoretical predictions
    dist = report['inventory_distribution']
    p12 = dist.get((1, 2), 0)
    p23 = dist.get((2, 3), 0)
    p31 = dist.get((3, 1), 0)
    
    print(f"\nInventory Distribution:")
    print(f"p12 = {p12:.3f} (expected: ~1.0)")
    print(f"p23 = {p23:.3f} (expected: ~0.5)")
    print(f"p31 = {p31:.3f} (expected: ~1.0)")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Inventory distribution
    plt.subplot(1, 2, 1)
    plot_inventory_distribution(economy)
    plt.title("Model A Fundamental: Inventory Distribution")
    
    # Trading matrix
    plt.subplot(1, 2, 2)
    plot_trading_matrix(economy)
    
    plt.tight_layout()
    plt.savefig('experiment1_model_a_fundamental.png', dpi=150)
    print("\nFigure saved as: experiment1_model_a_fundamental.png")
    
    return economy, analyzer


def experiment_2_model_a_speculative():
    """
    Replicate Model A speculative equilibrium (Theorem 1b).
    Type I agents speculate by trading good 2 for good 3.
    Both goods 1 and 3 serve as commodity monies.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Model A Speculative Equilibrium")
    print("="*70)
    
    # Set parameters to ensure speculative equilibrium
    # c13 - c12 < (sqrt(2) - 1)*beta*u1
    storage_costs = {
        (1, 1): 0.5, (1, 2): 1.0, (1, 3): 1.4,  # Small difference
        (2, 1): 0.5, (2, 2): 1.0, (2, 3): 1.4,
        (3, 1): 0.5, (3, 2): 1.0, (3, 3): 1.4
    }
    
    economy = KWEconomy(
        model_type='A',
        num_agents=300,
        storage_costs=storage_costs,
        utilities={1: 10.0, 2: 10.0, 3: 10.0},
        beta=0.95,
        strategy_name='speculative',
        random_seed=42
    )
    
    results = economy.run_simulation(num_periods=2000, burn_in=500)
    
    analyzer = KWAnalyzer(economy)
    report = analyzer.generate_report()
    
    print(f"\nEquilibrium type: {report['equilibrium_type']}")
    print(f"Media of exchange: {report['media_of_exchange']}")
    
    # Check theoretical predictions
    dist = report['inventory_distribution']
    p12 = dist.get((1, 2), 0)
    p13 = dist.get((1, 3), 0)
    
    print(f"\nInventory Distribution:")
    print(f"p12 = {p12:.3f} (expected: ~0.35)")
    print(f"p13 = {p13:.3f} (expected: ~0.65)")
    print("\nNote: Type I agents now hold good 3 more often,")
    print("demonstrating speculation despite higher storage cost.")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plot_inventory_distribution(economy)
    plt.title("Model A Speculative: Inventory Distribution")
    
    plt.subplot(1, 2, 2)
    plot_trading_matrix(economy)
    
    plt.tight_layout()
    plt.savefig('experiment2_model_a_speculative.png', dpi=150)
    print("\nFigure saved as: experiment2_model_a_speculative.png")
    
    return economy, analyzer


def experiment_3_model_b_multiple_equilibria():
    """
    Replicate Model B showing multiple equilibria coexistence (Theorem 2).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Model B Multiple Equilibria")
    print("="*70)
    
    storage_costs = {
        (1, 1): 0.5, (1, 2): 1.0, (1, 3): 1.5,
        (2, 1): 0.5, (2, 2): 1.0, (2, 3): 1.5,
        (3, 1): 0.5, (3, 2): 1.0, (3, 3): 1.5
    }
    
    # Run fundamental equilibrium
    print("\n--- Fundamental Equilibrium ---")
    economy_fund = KWEconomy(
        model_type='B',
        num_agents=300,
        storage_costs=storage_costs,
        strategy_name='fundamental',
        random_seed=42
    )
    economy_fund.run_simulation(2000, burn_in=500)
    analyzer_fund = KWAnalyzer(economy_fund)
    report_fund = analyzer_fund.generate_report()
    
    print(f"Equilibrium: {report_fund['equilibrium_type']}")
    print(f"Media of exchange: {report_fund['media_of_exchange']}")
    
    # Run speculative equilibrium
    print("\n--- Speculative Equilibrium ---")
    economy_spec = KWEconomy(
        model_type='B',
        num_agents=300,
        storage_costs=storage_costs,
        strategy_name='speculative',
        random_seed=42
    )
    economy_spec.run_simulation(2000, burn_in=500)
    analyzer_spec = KWAnalyzer(economy_spec)
    report_spec = analyzer_spec.generate_report()
    
    print(f"Equilibrium: {report_spec['equilibrium_type']}")
    print(f"Media of exchange: {report_spec['media_of_exchange']}")
    
    # Compare welfare
    print("\n--- Welfare Comparison ---")
    welfare_comparison = {
        'Fundamental': report_fund['welfare_by_type'],
        'Speculative': report_spec['welfare_by_type']
    }
    
    fig = plot_welfare_comparison(welfare_comparison)
    plt.savefig('experiment3_welfare_comparison.png', dpi=150)
    print("\nFigure saved as: experiment3_welfare_comparison.png")
    
    print("\nNote: Equilibria are NOT Pareto-comparable:")
    print("Type I prefers speculative, Types II & III prefer fundamental")
    
    return economy_fund, economy_spec


def experiment_4_fiat_money():
    """
    Replicate fiat money equilibrium (Theorem 3).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Fiat Money Equilibrium")
    print("="*70)
    
    storage_costs = {
        (1, 0): 0.0, (1, 1): 0.5, (1, 2): 1.0, (1, 3): 2.0,
        (2, 0): 0.0, (2, 1): 0.5, (2, 2): 1.0, (2, 3): 2.0,
        (3, 0): 0.0, (3, 1): 0.5, (3, 2): 1.0, (3, 3): 2.0
    }
    
    # Test different levels of fiat money
    fiat_proportions = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    results_by_proportion = {}
    
    for S in fiat_proportions:
        print(f"\n--- Real Balances S = {S:.1f} ---")
        
        economy = KWEconomy(
            model_type='A',
            num_agents=300,
            storage_costs=storage_costs,
            strategy_name='fiat',
            fiat_money=True,
            fiat_proportion=S,
            random_seed=42
        )
        
        economy.run_simulation(2000, burn_in=500)
        analyzer = KWAnalyzer(economy)
        report = analyzer.generate_report()
        
        results_by_proportion[S] = report
        
        print(f"Fiat acceptability: {report['acceptabilities'].get(0, 0):.3f}")
        print(f"Fiat velocity: {report['velocities'].get(0, 0):.3f}")
        print(f"General media: {report['general_media']}")
    
    # Check welfare improvement from introducing fiat money
    print("\n--- Welfare Analysis ---")
    print("Comparing S=0 (no fiat) vs S=0.3 (some fiat):")
    
    for agent_type in [1, 2, 3]:
        w0 = results_by_proportion[0.0]['welfare_by_type'][agent_type]
        w3 = results_by_proportion[0.4]['welfare_by_type'][agent_type]
        change = ((w3 - w0) / abs(w0)) * 100 if w0 != 0 else 0
        
        print(f"Type {agent_type}: {w0:.2f} -> {w3:.2f} ({change:+.1f}%)")
    
    print("\nNote: Introducing fiat money can improve welfare")
    print("by reducing inefficient storage of real commodities.")
    
    return results_by_proportion


def experiment_5_figure_9_replication():
    """
    Replicate Figure 9: velocity, acceptability, stocks vs real balances.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Replicate Figure 9")
    print("="*70)
    
    storage_costs = {
        (1, 0): 0.0, (1, 1): 0.5, (1, 2): 1.0, (1, 3): 2.0,
        (2, 0): 0.0, (2, 1): 0.5, (2, 2): 1.0, (2, 3): 2.0,
        (3, 0): 0.0, (3, 1): 0.5, (3, 2): 1.0, (3, 3): 2.0
    }
    
    # Sweep over real balances
    real_balances = np.linspace(0, 0.9, 10)
    results = {}
    
    print("\nRunning simulations for different real balances...")
    for i, S in enumerate(real_balances):
        print(f"Progress: {i+1}/{len(real_balances)} (S={S:.2f})")
        
        economy = KWEconomy(
            model_type='A',
            num_agents=300,
            storage_costs=storage_costs,
            strategy_name='fiat',
            fiat_money=True,
            fiat_proportion=S,
            random_seed=42
        )
        
        economy.run_simulation(1500, burn_in=300)
        analyzer = KWAnalyzer(economy)
        
        results[S] = {
            'stocks': analyzer.compute_all_stocks(),
            'velocities': analyzer.compute_all_velocities(),
            'acceptabilities': analyzer.compute_all_acceptabilities()
        }
    
    # Create Figure 9 style plot
    print("\nCreating Figure 9 style visualization...")
    fig = create_figure_9_style(results, figsize=(14, 12))
    plt.savefig('experiment5_figure9_replication.png', dpi=150)
    print("\nFigure saved as: experiment5_figure9_replication.png")
    
    print("\nKey observations from Figure 9:")
    print("- Velocity is NOT a good indicator of moneyness")
    print("- Good 3 has high velocity despite not being money")
    print("- Acceptability is a much better measure")
    print("- Fiat money has acceptability = 1 (general medium)")
    
    return results


def run_all_experiments():
    """Run all experiments to replicate paper results."""
    print("\n" + "#"*70)
    print("# KIYOTAKI-WRIGHT MODEL: REPLICATION OF MAIN RESULTS")
    print("#"*70)
    
    # Experiment 1: Model A Fundamental
    eco1, ana1 = experiment_1_model_a_fundamental()
    
    # Experiment 2: Model A Speculative
    eco2, ana2 = experiment_2_model_a_speculative()
    
    # Experiment 3: Model B Multiple Equilibria
    eco3_f, eco3_s = experiment_3_model_b_multiple_equilibria()
    
    # Experiment 4: Fiat Money
    results4 = experiment_4_fiat_money()
    
    # Experiment 5: Figure 9
    results5 = experiment_5_figure_9_replication()
    
    print("\n" + "#"*70)
    print("# ALL EXPERIMENTS COMPLETED")
    print("#"*70)
    print("\nResults saved to:")
    print("  - experiment1_model_a_fundamental.png")
    print("  - experiment2_model_a_speculative.png")
    print("  - experiment3_welfare_comparison.png")
    print("  - experiment5_figure9_replication.png")


if __name__ == "__main__":
    run_all_experiments()
    plt.show()
