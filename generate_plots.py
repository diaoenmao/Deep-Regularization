"""Generate Accuracy-Sparsity curve plots for the experiment results."""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('results/figures', exist_ok=True)

# Load results
with open('results/metrics/full_experiment_20260205_025759.json', 'r') as f:
    results = json.load(f)

# Color schemes
COLORS = {
    'Magnitude': '#1f77b4',
    'First-Order': '#ff7f0e',
    'Second-Order': '#2ca02c',
    'First+Second-Order': '#d62728',
}

MARKERS = {
    'Magnitude': 'o',
    'First-Order': 's',
    'Second-Order': '^',
    'First+Second-Order': 'D',
}

def get_sparsity(remaining_weights):
    """Convert remaining weights to sparsity percentage."""
    return [(1 - w) * 100 for w in remaining_weights]


def plot_method_comparison(method_name, title, filename):
    """Plot accuracy vs sparsity for a single method with all score types."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for entry in results:
        if entry['class_name'] == method_name:
            score_name = entry['score_name']
            sparsity = get_sparsity(entry['remaining_weights'])
            accuracy = entry['accuracy']

            # Sort by sparsity for proper line plotting
            sorted_pairs = sorted(zip(sparsity, accuracy))
            sparsity_sorted = [p[0] for p in sorted_pairs]
            accuracy_sorted = [p[1] for p in sorted_pairs]

            ax.plot(sparsity_sorted, accuracy_sorted,
                   marker=MARKERS.get(score_name, 'o'),
                   color=COLORS.get(score_name, 'gray'),
                   label=score_name, linewidth=2, markersize=8)

    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f'results/figures/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: results/figures/{filename}')


def plot_admm_comparison():
    """Plot all ADMM methods comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = [
        ('ADMM_Adam_Global', 'ADMM Global'),
        ('ADMM_Adam_Layer', 'ADMM Layer'),
        ('ADMM_Adam_Neuron', 'ADMM Neuron'),
    ]

    for ax, (method_name, title) in zip(axes, methods):
        for entry in results:
            if entry['class_name'] == method_name:
                score_name = entry['score_name']
                sparsity = get_sparsity(entry['remaining_weights'])
                accuracy = entry['accuracy']

                sorted_pairs = sorted(zip(sparsity, accuracy))
                sparsity_sorted = [p[0] for p in sorted_pairs]
                accuracy_sorted = [p[1] for p in sorted_pairs]

                ax.plot(sparsity_sorted, accuracy_sorted,
                       marker=MARKERS.get(score_name, 'o'),
                       color=COLORS.get(score_name, 'gray'),
                       label=score_name, linewidth=2, markersize=6)

        ax.set_xlabel('Sparsity (%)', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 100)

    plt.suptitle('ADMM Methods: Accuracy vs Sparsity', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/figures/accuracy_sparsity_admm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/figures/accuracy_sparsity_admm.png')


def plot_ppercent_comparison():
    """Plot all Ppercent methods comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = [
        ('Ppercent_Adam_Global', 'Ppercent Global'),
        ('Ppercent_Adam_Layer', 'Ppercent Layer'),
        ('Ppercent_Adam_Neuron', 'Ppercent Neuron'),
    ]

    for ax, (method_name, title) in zip(axes, methods):
        for entry in results:
            if entry['class_name'] == method_name:
                score_name = entry['score_name']
                sparsity = get_sparsity(entry['remaining_weights'])
                accuracy = entry['accuracy']

                sorted_pairs = sorted(zip(sparsity, accuracy))
                sparsity_sorted = [p[0] for p in sorted_pairs]
                accuracy_sorted = [p[1] for p in sorted_pairs]

                ax.plot(sparsity_sorted, accuracy_sorted,
                       marker=MARKERS.get(score_name, 'o'),
                       color=COLORS.get(score_name, 'gray'),
                       label=score_name, linewidth=2, markersize=6)

        ax.set_xlabel('Sparsity (%)', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)
        ax.set_ylim(85, 100)

    plt.suptitle('Ppercent Methods: Accuracy vs Sparsity', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/figures/accuracy_sparsity_ppercent.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/figures/accuracy_sparsity_ppercent.png')


def plot_lasso_comparison():
    """Plot all Lasso methods comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = [
        ('Lasso_Adam_Global', 'Lasso Global'),
        ('Lasso_Adam_Layer', 'Lasso Layer'),
        ('Lasso_Adam_Neuron', 'Lasso Neuron'),
    ]

    for ax, (method_name, title) in zip(axes, methods):
        for entry in results:
            if entry['class_name'] == method_name:
                score_name = entry['score_name']
                sparsity = get_sparsity(entry['remaining_weights'])
                accuracy = entry['accuracy']

                sorted_pairs = sorted(zip(sparsity, accuracy))
                sparsity_sorted = [p[0] for p in sorted_pairs]
                accuracy_sorted = [p[1] for p in sorted_pairs]

                ax.plot(sparsity_sorted, accuracy_sorted,
                       marker=MARKERS.get(score_name, 'o'),
                       color=COLORS.get(score_name, 'gray'),
                       label=score_name, linewidth=2, markersize=6)

        ax.set_xlabel('Sparsity (%)', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 100)

    plt.suptitle('Lasso Methods: Accuracy vs Sparsity', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/figures/accuracy_sparsity_lasso.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/figures/accuracy_sparsity_lasso.png')


def plot_best_methods():
    """Plot best configuration from each method family."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Best configurations (magnitude score for consistency)
    best_configs = [
        ('ADMM_Adam_Global', 'Magnitude', 'ADMM Global', '#1f77b4', 'o'),
        ('ADMM_Adam_Layer', 'Magnitude', 'ADMM Layer', '#aec7e8', 's'),
        ('ADMM_Adam_Neuron', 'Magnitude', 'ADMM Neuron', '#ff7f0e', '^'),
        ('Ppercent_Adam_Global', 'Magnitude', 'Ppercent Global', '#2ca02c', 'D'),
        ('Ppercent_Adam_Neuron', 'Magnitude', 'Ppercent Neuron', '#98df8a', 'v'),
        ('Lasso_Adam_Global', 'Magnitude', 'Lasso Global', '#d62728', 'p'),
        ('Lasso_Adam_Neuron', 'Magnitude', 'Lasso Neuron', '#ff9896', 'h'),
    ]

    for method_name, score_name, label, color, marker in best_configs:
        for entry in results:
            if entry['class_name'] == method_name and entry['score_name'] == score_name:
                sparsity = get_sparsity(entry['remaining_weights'])
                accuracy = entry['accuracy']

                # Filter out collapsed points (accuracy < 20%)
                valid_pairs = [(s, a) for s, a in zip(sparsity, accuracy) if a > 20]
                if valid_pairs:
                    sorted_pairs = sorted(valid_pairs)
                    sparsity_sorted = [p[0] for p in sorted_pairs]
                    accuracy_sorted = [p[1] for p in sorted_pairs]

                    ax.plot(sparsity_sorted, accuracy_sorted,
                           marker=marker, color=color, label=label,
                           linewidth=2.5, markersize=10)

    ax.set_xlabel('Sparsity (%)', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Best Methods Comparison (Magnitude Score)', fontsize=16, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(80, 100)

    # Add annotation for best compression point
    ax.annotate('94.66% @ 99% sparse\n(100x compression)',
                xy=(99, 94.66), xytext=(75, 88),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig('results/figures/accuracy_sparsity_best.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/figures/accuracy_sparsity_best.png')


def plot_all_methods():
    """Plot all methods in a single comprehensive figure."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    all_methods = [
        [('ADMM_Adam_Global', 'ADMM Global'), ('ADMM_Adam_Layer', 'ADMM Layer'), ('ADMM_Adam_Neuron', 'ADMM Neuron')],
        [('Ppercent_Adam_Global', 'Ppercent Global'), ('Ppercent_Adam_Layer', 'Ppercent Layer'), ('Ppercent_Adam_Neuron', 'Ppercent Neuron')],
        [('Lasso_Adam_Global', 'Lasso Global'), ('Lasso_Adam_Layer', 'Lasso Layer'), ('Lasso_Adam_Neuron', 'Lasso Neuron')],
    ]

    for row_idx, row_methods in enumerate(all_methods):
        for col_idx, (method_name, title) in enumerate(row_methods):
            ax = axes[row_idx, col_idx]

            for entry in results:
                if entry['class_name'] == method_name:
                    score_name = entry['score_name']
                    sparsity = get_sparsity(entry['remaining_weights'])
                    accuracy = entry['accuracy']

                    sorted_pairs = sorted(zip(sparsity, accuracy))
                    sparsity_sorted = [p[0] for p in sorted_pairs]
                    accuracy_sorted = [p[1] for p in sorted_pairs]

                    ax.plot(sparsity_sorted, accuracy_sorted,
                           marker=MARKERS.get(score_name, 'o'),
                           color=COLORS.get(score_name, 'gray'),
                           label=score_name, linewidth=1.5, markersize=5)

            ax.set_xlabel('Sparsity (%)', fontsize=10)
            ax.set_ylabel('Accuracy (%)', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(loc='lower left', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 105)

            # Adjust y-axis based on method
            if 'Ppercent' in method_name:
                ax.set_ylim(85, 100)
            else:
                ax.set_ylim(0, 100)

    plt.suptitle('All Methods: Accuracy vs Sparsity Comparison', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('results/figures/accuracy_sparsity_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/figures/accuracy_sparsity_all.png')


def plot_pareto_frontier():
    """Plot Pareto frontier of best accuracy-sparsity tradeoffs."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Collect all points
    all_points = []
    for entry in results:
        sparsity = get_sparsity(entry['remaining_weights'])
        accuracy = entry['accuracy']
        method = entry['class_name']
        score = entry['score_name']

        for s, a in zip(sparsity, accuracy):
            if a > 20:  # Filter collapsed models
                all_points.append((s, a, method, score))

    # Find Pareto frontier
    pareto_points = []
    for point in all_points:
        s, a, m, sc = point
        is_dominated = False
        for other in all_points:
            os, oa, _, _ = other
            # A point is dominated if another has higher accuracy AND higher sparsity
            if os >= s and oa > a:
                is_dominated = True
                break
            if os > s and oa >= a:
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append(point)

    # Sort Pareto points by sparsity
    pareto_points.sort(key=lambda x: x[0])

    # Plot all points (faded)
    for entry in results:
        sparsity = get_sparsity(entry['remaining_weights'])
        accuracy = entry['accuracy']
        valid_pairs = [(s, a) for s, a in zip(sparsity, accuracy) if a > 20]
        if valid_pairs:
            s_vals = [p[0] for p in valid_pairs]
            a_vals = [p[1] for p in valid_pairs]
            ax.scatter(s_vals, a_vals, alpha=0.2, s=30, c='gray')

    # Plot Pareto frontier
    pareto_s = [p[0] for p in pareto_points]
    pareto_a = [p[1] for p in pareto_points]
    ax.plot(pareto_s, pareto_a, 'r-', linewidth=3, label='Pareto Frontier', zorder=5)
    ax.scatter(pareto_s, pareto_a, c='red', s=100, zorder=6, edgecolors='black')

    # Annotate key points
    key_points = [
        (35.1, 97.67, 'Ppercent (97.67%)'),
        (52.8, 95.71, 'ADMM Global'),
        (71.3, 95.66, 'ADMM Neuron'),
        (99.0, 94.66, 'ADMM Neuron\n(100x compression)'),
    ]

    for s, a, label in key_points:
        ax.annotate(label, xy=(s, a), xytext=(s+5, a-3),
                   fontsize=9, ha='left',
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=0.5))

    ax.set_xlabel('Sparsity (%)', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Pareto Frontier: Accuracy vs Sparsity', fontsize=16, fontweight='bold')
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(80, 100)

    plt.tight_layout()
    plt.savefig('results/figures/pareto_frontier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/figures/pareto_frontier.png')


if __name__ == '__main__':
    print('Generating plots...\n')

    # Individual method plots
    plot_method_comparison('ADMM_Adam_Global', 'ADMM Global: Accuracy vs Sparsity', 'admm_global.png')
    plot_method_comparison('ADMM_Adam_Layer', 'ADMM Layer: Accuracy vs Sparsity', 'admm_layer.png')
    plot_method_comparison('ADMM_Adam_Neuron', 'ADMM Neuron: Accuracy vs Sparsity', 'admm_neuron.png')
    plot_method_comparison('Ppercent_Adam_Global', 'Ppercent Global: Accuracy vs Sparsity', 'ppercent_global.png')
    plot_method_comparison('Lasso_Adam_Global', 'Lasso Global: Accuracy vs Sparsity', 'lasso_global.png')
    plot_method_comparison('Lasso_Adam_Neuron', 'Lasso Neuron: Accuracy vs Sparsity', 'lasso_neuron.png')

    # Comparison plots
    plot_admm_comparison()
    plot_ppercent_comparison()
    plot_lasso_comparison()
    plot_best_methods()
    plot_all_methods()
    plot_pareto_frontier()

    print('\nAll plots generated successfully!')
