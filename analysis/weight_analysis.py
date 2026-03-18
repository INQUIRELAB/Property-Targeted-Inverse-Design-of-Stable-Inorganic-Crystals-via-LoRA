#!/usr/bin/env python3
"""
Analysis 1: Singular Value Decomposition of Weight Updates

This script analyzes the rank structure of LoRA weight updates
and compares with the full FiLM model changes.

Key Questions:
1. What are the singular values of LoRA updates (B @ A)?
2. What is the effective rank of LoRA adaptations?
3. How much information is compressed into rank r?
4. For FiLM: what is the distribution of weight changes?

Usage:
    python 1_analyze_weight_updates.py \
        --lora_model path/to/lora_rank16.ckpt \
        --film_model path/to/film_adapter.ckpt \
        --base_model path/to/mp_20_base.ckpt \
        --output_dir svd_analysis
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import OrderedDict

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


class WeightUpdateAnalyzer:
    """Analyze weight updates using Singular Value Decomposition."""
    
    def __init__(self, lora_model_path, film_model_path, base_model_path):
        """Initialize analyzer with model paths."""
        print("🔧 Loading models...")
        self.lora_checkpoint = torch.load(lora_model_path, map_location='cpu', weights_only=False)
        self.film_checkpoint = torch.load(film_model_path, map_location='cpu', weights_only=False)
        self.base_checkpoint = torch.load(base_model_path, map_location='cpu', weights_only=False)
        
        print(f"✅ LoRA model loaded")
        print(f"✅ FiLM model loaded")
        print(f"✅ Base model loaded")
    
    def extract_lora_updates(self):
        """Extract ΔW = B @ A from LoRA layers."""
        print("\n📊 Extracting LoRA weight updates...")
        updates = OrderedDict()
        
        state_dict = self.lora_checkpoint.get('state_dict', self.lora_checkpoint)
        
        # Find LoRA layers
        lora_A_keys = [k for k in state_dict.keys() if 'lora_A' in k and 'weight' in k]
        
        for a_key in lora_A_keys:
            # Get corresponding B key
            b_key = a_key.replace('lora_A', 'lora_B')
            
            if b_key not in state_dict:
                continue
            
            try:
                # Extract A and B matrices
                A = state_dict[a_key].cpu().numpy()
                B = state_dict[b_key].cpu().numpy()
                
                # Compute low-rank update: ΔW = B @ A
                delta_W = B @ A
                
                # Get layer name
                layer_name = a_key.replace('.lora_A.weight', '').replace('model.', '')
                
                updates[layer_name] = {
                    'delta_W': delta_W,
                    'A': A,
                    'B': B,
                    'rank': A.shape[0],
                    'shape': delta_W.shape
                }
                
                print(f"  ✓ {layer_name}: rank={A.shape[0]}, shape={delta_W.shape}")
            except Exception as e:
                print(f"  ✗ Error processing {a_key}: {e}")
                continue
        
        print(f"\n✅ Extracted {len(updates)} LoRA layers")
        return updates
    
    def extract_film_updates(self):
        """
        Extract weight changes from FiLM model.
        
        Note: FiLM doesn't have equivalent LoRA adapter layers.
        Instead, we analyze the overall distribution of weight changes
        across all modified layers.
        """
        print("\n📊 Extracting FiLM weight changes...")
        updates = OrderedDict()
        
        film_state = self.film_checkpoint.get('state_dict', self.film_checkpoint)
        base_state = self.base_checkpoint.get('state_dict', self.base_checkpoint)
        
        # Find matching weight parameters
        total_film_params = 0
        total_changes = 0
        
        for key in film_state.keys():
            if 'weight' not in key:
                continue
            
            if key not in base_state:
                continue
            
            try:
                film_weight = film_state[key].cpu().numpy()
                base_weight = base_state[key].cpu().numpy()
                
                if film_weight.shape != base_weight.shape:
                    continue
                
                delta_W = film_weight - base_weight
                
                # Get layer name
                layer_name = key.replace('.weight', '').replace('model.', '')
                
                updates[layer_name] = {
                    'delta_W': delta_W,
                    'shape': delta_W.shape
                }
                
                total_film_params += delta_W.size
                total_changes += np.sum(np.abs(delta_W) > 1e-8)
                
            except Exception as e:
                continue
        
        print(f"\n✅ Extracted {len(updates)} FiLM layers")
        print(f"   Total parameters: {total_film_params:,}")
        print(f"   Parameters changed: {total_changes:,} ({total_changes/total_film_params*100:.2f}%)")
        
        return updates
    
    def compute_svd(self, weight_matrix):
        """Compute Singular Value Decomposition."""
        # Ensure 2D matrix
        if len(weight_matrix.shape) > 2:
            weight_matrix = weight_matrix.reshape(weight_matrix.shape[0], -1)
        
        try:
            U, S, Vh = np.linalg.svd(weight_matrix, full_matrices=False)
            return U, S, Vh
        except Exception as e:
            print(f"    ⚠️  SVD failed: {e}")
            return None, None, None
    
    def analyze_lora_singular_values(self, lora_updates):
        """Analyze singular values of LoRA updates."""
        print("\n🔍 Analyzing LoRA singular values...")
        results = OrderedDict()
        
        for layer_name, update_info in lora_updates.items():
            print(f"\n  Layer: {layer_name}")
            
            delta_W = update_info['delta_W']
            rank = update_info['rank']
            
            # Ensure 2D
            if len(delta_W.shape) > 2:
                original_shape = delta_W.shape
                delta_W = delta_W.reshape(delta_W.shape[0], -1)
                print(f"    Reshaped from {original_shape} to {delta_W.shape}")
            
            U, S, Vh = self.compute_svd(delta_W)
            
            if S is None:
                continue
            
            # Compute metrics
            total_energy = np.sum(S**2)
            rank_energy = np.sum(S[:rank]**2) if len(S) >= rank else np.sum(S**2)
            energy_ratio = rank_energy / total_energy if total_energy > 1e-10 else 1.0
            
            # Skip zero-energy layers (uninitialized or unused)
            if total_energy < 1e-10:
                print(f"    ⚠️  Skipping (zero energy, likely uninitialized)")
                continue
            
            # Effective rank (number of components capturing 90% energy)
            cumsum = np.cumsum(S**2) / total_energy
            effective_rank = np.argmax(cumsum >= 0.9) + 1 if np.any(cumsum >= 0.9) else len(S)
            
            results[layer_name] = {
                'singular_values': S,
                'rank': rank,
                'total_energy': total_energy,
                'rank_energy': rank_energy,
                'energy_ratio': energy_ratio,
                'effective_rank_90': effective_rank,
                'U': U,
                'Vh': Vh
            }
            
            print(f"    ✓ Rank: {rank}")
            print(f"    ✓ Total energy: {total_energy:.3e}")
            print(f"    ✓ Energy in rank {rank}: {energy_ratio*100:.2f}%")
            print(f"    ✓ Effective rank (90% energy): {effective_rank}")
        
        print(f"\n✅ Analyzed {len(results)} LoRA layers")
        return results
    
    def analyze_film_weight_distribution(self, film_updates):
        """Analyze distribution of FiLM weight changes."""
        print("\n🔍 Analyzing FiLM weight distribution...")
        
        all_changes = []
        layer_stats = OrderedDict()
        
        for layer_name, update_info in film_updates.items():
            delta_W = update_info['delta_W']
            
            # Flatten
            changes_flat = delta_W.flatten()
            all_changes.append(changes_flat)
            
            # Stats for this layer
            layer_stats[layer_name] = {
                'mean_abs_change': np.mean(np.abs(changes_flat)),
                'std_change': np.std(changes_flat),
                'max_abs_change': np.max(np.abs(changes_flat)),
                'fraction_changed': np.mean(np.abs(changes_flat) > 1e-6)
            }
        
        # Aggregate statistics
        all_changes_flat = np.concatenate(all_changes)
        
        overall_stats = {
            'total_parameters': len(all_changes_flat),
            'mean_abs_change': np.mean(np.abs(all_changes_flat)),
            'median_abs_change': np.median(np.abs(all_changes_flat)),
            'std_change': np.std(all_changes_flat),
            'max_abs_change': np.max(np.abs(all_changes_flat)),
            'fraction_changed': np.mean(np.abs(all_changes_flat) > 1e-6)
        }
        
        print(f"\n  Total parameters analyzed: {overall_stats['total_parameters']:,}")
        print(f"  Mean absolute change: {overall_stats['mean_abs_change']:.3e}")
        print(f"  Median absolute change: {overall_stats['median_abs_change']:.3e}")
        print(f"  Fraction changed: {overall_stats['fraction_changed']*100:.2f}%")
        
        return overall_stats, layer_stats, all_changes_flat
    
    def visualize_lora_results(self, lora_results, output_dir):
        """Visualize LoRA SVD results."""
        print("\n📊 Creating LoRA visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Select layers to plot (up to 4 for clarity)
        layers_to_plot = list(lora_results.keys())[:4]
        
        if not layers_to_plot:
            print("  ⚠️  No layers to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, layer_name in enumerate(layers_to_plot):
            ax = axes[idx]
            results = lora_results[layer_name]
            
            rank = results['rank']
            S = results['singular_values']
            energy_ratio = results['energy_ratio']
            effective_rank = results['effective_rank_90']
            
            # Plot singular values
            max_comp = min(len(S), 50)
            
            ax.semilogy(range(max_comp), S[:max_comp], 'o-', 
                       color='steelblue', linewidth=2, markersize=6,
                       label=f'Singular values')
            ax.axvline(rank, color='red', linestyle='--', linewidth=2.5,
                      label=f'LoRA rank ({rank})')
            ax.axvline(effective_rank, color='green', linestyle=':', linewidth=2,
                      label=f'90% energy ({effective_rank})')
            
            # Shade captured energy
            ax.fill_between(range(rank), 1e-10, S[:rank],
                           alpha=0.2, color='orange')
            
            ax.set_xlabel('Component Index', fontsize=11)
            ax.set_ylabel('Singular Value', fontsize=11)
            ax.set_title(f'Layer {idx+1}: {energy_ratio*100:.1f}% energy in rank {rank}',
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_path / 'lora_singular_values.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
        
        # Summary plot
        self._create_lora_summary(lora_results, output_path)
    
    def _create_lora_summary(self, lora_results, output_path):
        """Create summary plot of LoRA energy ratios."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        layer_names = list(lora_results.keys())
        ranks = [lora_results[name]['rank'] for name in layer_names]
        energy_ratios = [lora_results[name]['energy_ratio'] for name in layer_names]
        effective_ranks = [lora_results[name]['effective_rank_90'] for name in layer_names]
        
        # Plot 1: Energy ratios
        bars = ax1.bar(range(len(layer_names)), energy_ratios,
                      color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axhline(0.9, color='red', linestyle='--', linewidth=2,
                   label='90% threshold')
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Energy Ratio', fontsize=12)
        ax1.set_title(f'Energy Captured by LoRA Rank ({ranks[0]})', 
                     fontsize=13, fontweight='bold')
        ax1.set_ylim([0, 1.05])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Effective rank vs. LoRA rank
        ax2.scatter(ranks, effective_ranks, s=100, color='purple',
                   alpha=0.7, edgecolors='black', linewidths=2)
        ax2.plot([0, max(ranks)+5], [0, max(ranks)+5], 'k--',
                linewidth=2, label='Perfect match')
        ax2.set_xlabel('LoRA Rank', fontsize=12)
        ax2.set_ylabel('Effective Rank (90% energy)', fontsize=12)
        ax2.set_title('LoRA Rank vs. Effective Rank', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_path / 'lora_energy_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    def visualize_film_distribution(self, all_changes_flat, output_dir):
        """Visualize FiLM weight change distribution."""
        print("\n📊 Creating FiLM weight change distribution...")
        
        output_path = Path(output_dir)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(all_changes_flat, bins=100, color='coral', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Weight Change (ΔW)', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Distribution of FiLM Weight Changes', fontsize=13, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_abs_changes = np.sort(np.abs(all_changes_flat))
        cumsum = np.cumsum(sorted_abs_changes**2)
        cumsum_normalized = cumsum / cumsum[-1]
        
        ax2.plot(range(len(cumsum_normalized)), cumsum_normalized,
                color='darkblue', linewidth=2)
        ax2.axhline(0.9, color='red', linestyle='--', linewidth=2,
                   label='90% threshold')
        ax2.set_xlabel('Parameter Index (sorted by |ΔW|)', fontsize=12)
        ax2.set_ylabel('Cumulative Energy Fraction', fontsize=12)
        ax2.set_title('FiLM: Cumulative Energy Distribution', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_path / 'film_weight_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    def create_comparison_figure(self, lora_results, film_stats, output_dir):
        """Create comparison figure between LoRA and FiLM."""
        print("\n📊 Creating LoRA vs. FiLM comparison...")
        
        output_path = Path(output_dir)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Calculate average LoRA behavior
        avg_energy_ratio = np.mean([r['energy_ratio'] for r in lora_results.values()])
        avg_rank = np.mean([r['rank'] for r in lora_results.values()])
        avg_effective_rank = np.mean([r['effective_rank_90'] for r in lora_results.values()])
        
        # Plot comparison
        categories = ['Energy Captured\nby Rank', 'Parameter\nEfficiency', 'Information\nCompression']
        lora_values = [
            avg_energy_ratio * 100,  # Convert to percentage
            100 - (avg_rank / avg_effective_rank * 100),  # How much compression
            (1 - avg_energy_ratio) * 100  # Information loss
        ]
        film_values = [
            100,  # FiLM captures 100% (no rank constraint)
            0,    # FiLM has no compression
            0     # FiLM has no information loss
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, lora_values, width,
                      label=f'LoRA (Rank {int(avg_rank)})', 
                      color='steelblue', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, film_values, width,
                      label='FiLM (Full)',
                      color='coral', alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
        ax.set_title('LoRA vs. FiLM: Information Content Comparison',
                    fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 110])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = output_path / 'lora_vs_film_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    def save_results(self, lora_results, film_stats, output_dir):
        """Save numerical results to JSON."""
        print("\n💾 Saving results...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare LoRA results for JSON
        lora_json = {}
        for layer_name, results in lora_results.items():
            lora_json[layer_name] = {
                'rank': int(results['rank']),
                'total_energy': float(results['total_energy']),
                'rank_energy': float(results['rank_energy']),
                'energy_ratio': float(results['energy_ratio']),
                'effective_rank_90': int(results['effective_rank_90']),
                'singular_values': results['singular_values'].tolist()[:50]  # First 50
            }
        
        # Save LoRA results
        lora_file = output_path / 'lora_svd_results.json'
        with open(lora_file, 'w') as f:
            json.dump(lora_json, f, indent=2)
        print(f"  ✓ Saved LoRA results: {lora_file}")
        
        # Save FiLM stats (convert numpy types to Python types)
        film_stats_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else int(v) if isinstance(v, np.integer) else v 
                          for k, v in film_stats.items()}
        
        film_file = output_path / 'film_stats.json'
        with open(film_file, 'w') as f:
            json.dump(film_stats_json, f, indent=2)
        print(f"  ✓ Saved FiLM stats: {film_file}")
    
    def print_summary(self, lora_results, film_stats):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("WEIGHT UPDATE ANALYSIS SUMMARY")
        print("="*70)
        
        print("\nLoRA Analysis:")
        print("-" * 40)
        
        energy_ratios = [r['energy_ratio'] for r in lora_results.values()]
        ranks = [r['rank'] for r in lora_results.values()]
        effective_ranks = [r['effective_rank_90'] for r in lora_results.values()]
        
        print(f"  Layers analyzed: {len(lora_results)}")
        print(f"  LoRA rank: {ranks[0]}")
        print(f"  Average energy captured: {np.mean(energy_ratios)*100:.2f}%")
        print(f"  Min energy captured: {np.min(energy_ratios)*100:.2f}%")
        print(f"  Max energy captured: {np.max(energy_ratios)*100:.2f}%")
        print(f"  Average effective rank (90% energy): {np.mean(effective_ranks):.1f}")
        
        print("\nFiLM Analysis:")
        print("-" * 40)
        print(f"  Total parameters: {film_stats['total_parameters']:,}")
        print(f"  Mean absolute change: {film_stats['mean_abs_change']:.3e}")
        print(f"  Fraction changed: {film_stats['fraction_changed']*100:.2f}%")
        
        print("\n" + "-"*70)
        print("INTERPRETATION")
        print("-"*70)
        
        avg_energy = np.mean(energy_ratios)
        if avg_energy >= 0.9:
            print("  ✅ LoRA captures >90% of energy on average")
            print("     → Low-rank approximation is reasonable")
        else:
            print(f"  ⚠️  LoRA captures {avg_energy*100:.1f}% of energy")
            print(f"     → {(1-avg_energy)*100:.1f}% information loss may impact diversity")
        
        if np.mean(effective_ranks) > np.mean(ranks) * 1.2:
            print(f"  ⚠️  Effective rank ({np.mean(effective_ranks):.1f}) > LoRA rank ({np.mean(ranks):.1f})")
            print("     → May benefit from higher rank")
        else:
            print(f"  ✅ LoRA rank matches effective rank closely")
        
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze weight updates via Singular Value Decomposition'
    )
    parser.add_argument('--lora_model', type=str, required=True,
                       help='Path to trained LoRA model checkpoint')
    parser.add_argument('--film_model', type=str, required=True,
                       help='Path to trained FiLM model checkpoint')
    parser.add_argument('--base_model', type=str, required=True,
                       help='Path to base pretrained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='svd_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SINGULAR VALUE DECOMPOSITION ANALYSIS")
    print("="*70)
    print(f"LoRA model: {args.lora_model}")
    print(f"FiLM model: {args.film_model}")
    print(f"Base model: {args.base_model}")
    print(f"Output: {args.output_dir}")
    print("="*70 + "\n")
    
    # Initialize analyzer
    analyzer = WeightUpdateAnalyzer(args.lora_model, args.film_model, args.base_model)
    
    # Extract updates
    lora_updates = analyzer.extract_lora_updates()
    film_updates = analyzer.extract_film_updates()
    
    if not lora_updates:
        print("\n❌ No LoRA layers found")
        return 1
    
    # Analyze LoRA singular values
    lora_results = analyzer.analyze_lora_singular_values(lora_updates)
    
    # Analyze FiLM weight distribution
    film_stats, _, all_changes_flat = analyzer.analyze_film_weight_distribution(film_updates)
    
    # Visualize
    analyzer.visualize_lora_results(lora_results, args.output_dir)
    analyzer.visualize_film_distribution(all_changes_flat, args.output_dir)
    analyzer.create_comparison_figure(lora_results, film_stats, args.output_dir)
    
    # Save results
    analyzer.save_results(lora_results, film_stats, args.output_dir)
    
    # Print summary
    analyzer.print_summary(lora_results, film_stats)
    
    print("✅ Analysis complete!\n")
    print("📊 Key Outputs:")
    print(f"   - lora_singular_values.png")
    print(f"   - lora_energy_summary.png")
    print(f"   - film_weight_distribution.png")
    print(f"   - lora_vs_film_comparison.png")
    print(f"   - lora_svd_results.json")
    print(f"   - film_stats.json\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
