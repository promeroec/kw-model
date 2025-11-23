"""
Simple example demonstrating the Kiyotaki-Wright model

Run this script to see the model in action:
    python example_simple.py
"""

import sys
sys.path.append('..')

from kw_model import KWEconomy, KWAnalyzer
from kw_model.visualization import plot_inventory_distribution
import matplotlib.pyplot as plt


def main():
    """Run a simple demonstration of the model."""
    
    print("\n" + "="*60)
    print("KIYOTAKI-WRIGHT MODEL - SIMPLE DEMONSTRATION")
    print("="*60)
    
    # Create economy
    print("\nCreating Model A economy with fundamental strategy...")
    economy = KWEconomy(
        model_type='A',
        num_agents=300,
        strategy_name='fundamental',
        beta=0.95,
        random_seed=42
    )
    
    print(f"✓ Created economy with {economy.num_agents} agents")
    print(f"  - {economy.num_agents//3} agents of each type")
    print(f"  - Model type: {economy.model_type}")
    print(f"  - Strategy: fundamental")
    
    # Run simulation
    print("\nRunning simulation...")
    print("  - Total periods: 1000")
    print("  - Burn-in periods: 200")
    
    results = economy.run_simulation(num_periods=1000, burn_in=200)
    
    print("✓ Simulation complete!")
    
    # Analyze results
    print("\nAnalyzing results...")
    analyzer = KWAnalyzer(economy)
    report = analyzer.generate_report()
    
    # Print key results
    print("\n" + "-"*60)
    print("KEY RESULTS")
    print("-"*60)
    
    print(f"\nEquilibrium Type: {report['equilibrium_type']}")
    print(f"Media of Exchange: {report['media_of_exchange']}")
    print(f"General Media: {report['general_media']}")
    
    print("\nInventory Distribution (p_ij):")
    for agent_type in [1, 2, 3]:
        print(f"  Type {agent_type}:")
        for good in [1, 2, 3]:
            prop = report['inventory_distribution'].get((agent_type, good), 0)
            print(f"    Good {good}: {prop:.4f}")
    
    print("\nVelocities:")
    for good, velocity in report['velocities'].items():
        print(f"  Good {good}: {velocity:.4f}")
    
    print("\nAcceptabilities:")
    for good, accept in report['acceptabilities'].items():
        print(f"  Good {good}: {accept:.4f}")
    
    print("\nWelfare by Type:")
    for agent_type, welfare in report['welfare_by_type'].items():
        print(f"  Type {agent_type}: {welfare:.4f}")
    
    # Expected results for Model A fundamental
    print("\n" + "-"*60)
    print("THEORETICAL PREDICTIONS (Model A Fundamental)")
    print("-"*60)
    print("Expected inventory distribution:")
    print("  p12 ≈ 1.0 (Type I always holds good 2)")
    print("  p23 ≈ 0.5 (Type II holds goods 1 and 3 equally)")
    print("  p31 ≈ 1.0 (Type III always holds good 1)")
    print("\nExpected: Good 1 is the unique commodity money")
    
    # Visualize
    print("\n" + "-"*60)
    print("VISUALIZATION")
    print("-"*60)
    print("Generating plot...")
    
    fig, ax = plot_inventory_distribution(economy)
    plt.savefig('simple_example_result.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved as: simple_example_result.png")
    
    plt.show()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Try changing strategy_name to 'speculative'")
    print("2. Set fiat_money=True to test fiat currency")
    print("3. Run experiments/replicate_paper.py for full analysis")
    print("4. Check README.md for more examples")
    print("\n")


if __name__ == "__main__":
    main()
