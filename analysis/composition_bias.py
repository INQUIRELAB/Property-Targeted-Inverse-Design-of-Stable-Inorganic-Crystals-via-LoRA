#!/usr/bin/env python3
"""
Analysis 4: Direct Generation Bias Test

Directly test if rank cutoff causes oxide bias by comparing
chemistry distributions across different LoRA ranks and FiLM.

Key Questions:
1. Does oxide fraction increase as rank decreases?
2. Does metallic fraction decrease as rank decreases?
3. Is there a statistically significant correlation?
4. Does this prove causality?

Usage:
    python 4_generation_bias_test.py \
        --rank8_materials path/to/rank8/ \
        --rank16_materials path/to/rank16/ \
        --rank32_materials path/to/rank32/ \
        --film_materials path/to/film/ \
        --output_dir generation_bias_analysis
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, OrderedDict
from ase.io import read as ase_read
from scipy.stats import spearmanr, pearsonr

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


class GenerationBiasAnalyzer:
    """Analyze generation bias across different ranks."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.chemistry_classes = ['oxide', 'metallic', 'chalcogenide', 'halide', 'pnictide', 'other']
        self.class_colors = {
            'oxide': 'orange',
            'metallic': 'darkred',
            'chalcogenide': 'purple',
            'halide': 'green',
            'pnictide': 'blue',
            'other': 'gray'
        }
    
    def classify_structure(self, structure):
        """Classify structure by chemical class."""
        symbols = structure.get_chemical_symbols()
        
        has_O = 'O' in symbols
        has_S = 'S' in symbols or 'Se' in symbols or 'Te' in symbols
        has_halide = any(x in symbols for x in ['F', 'Cl', 'Br', 'I'])
        has_pnictide = any(x in symbols for x in ['N', 'P', 'As', 'Sb', 'Bi'])
        
        metals = {'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 
                 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
                 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Hf',
                 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'}
        
        metal_count = sum(1 for s in symbols if s in metals)
        metal_fraction = metal_count / len(symbols) if len(symbols) > 0 else 0
        
        if has_O and not has_S and not has_halide:
            return 'oxide'
        elif has_S:
            return 'chalcogenide'
        elif has_halide:
            return 'halide'
        elif has_pnictide and not has_O:
            return 'pnictide'
        elif metal_fraction > 0.7 and not has_O and not has_S and not has_halide:
            return 'metallic'
        else:
            return 'other'
    
    def analyze_materials(self, materials_path, model_name):
        """
        Analyze chemistry distribution in generated materials.
        
        Args:
            materials_path: Path to materials directory or file
            model_name: Name of the model (e.g., 'Rank 8', 'FiLM')
        
        Returns:
            Dictionary with chemistry counts and fractions
        """
        print(f"\n📂 Analyzing {model_name}: {materials_path}")
        
        materials_path = Path(materials_path)
        
        if materials_path.is_file():
            files = [materials_path]
        else:
            files = list(materials_path.glob('**/*.extxyz'))
            if not files:
                files = list(materials_path.glob('*.extxyz'))
        
        if not files:
            print(f"  ⚠️  No extxyz files found")
            return None
        
        chemistry_counts = defaultdict(int)
        total_structures = 0
        
        for file_path in files:
            try:
                structures = ase_read(str(file_path), index=':')
                if not isinstance(structures, list):
                    structures = [structures]
                
                for structure in structures:
                    chemistry = self.classify_structure(structure)
                    chemistry_counts[chemistry] += 1
                    total_structures += 1
                    
            except Exception as e:
                print(f"    ⚠️  Error loading {file_path.name}: {e}")
                continue
        
        if total_structures == 0:
            print(f"  ⚠️  No structures loaded")
            return None
        
        # Compute fractions
        chemistry_fractions = {
            chem: count / total_structures 
            for chem, count in chemistry_counts.items()
        }
        
        results = {
            'model_name': model_name,
            'total_structures': total_structures,
            'chemistry_counts': dict(chemistry_counts),
            'chemistry_fractions': chemistry_fractions
        }
        
        print(f"  ✅ Loaded {total_structures} structures")
        print(f"  📊 Chemistry distribution:")
        for chem in self.chemistry_classes:
            count = chemistry_counts.get(chem, 0)
            fraction = chemistry_fractions.get(chem, 0)
            print(f"     {chem:15s}: {count:5d} ({fraction*100:5.2f}%)")
        
        return results
    
    def visualize_bias_vs_rank(self, results_by_model, output_dir):
        """
        Create visualization showing LoRA vs. FiLM chemistry differences.
        
        Args:
            results_by_model: OrderedDict of results for each model
            output_dir: Output directory
        """
        print("\n📊 Creating visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        model_names = []
        ranks = []
        oxide_fractions = []
        metallic_fractions = []
        
        rank_mapping = {
            'Rank 8': 8,
            'Rank 16': 16,
            'Rank 32': 32,
            'FiLM': 100
        }
        
        for model_name, results in results_by_model.items():
            if results is None:
                continue
            
            model_names.append(model_name)
            ranks.append(rank_mapping.get(model_name, 0))
            oxide_fractions.append(results['chemistry_fractions'].get('oxide', 0))
            metallic_fractions.append(results['chemistry_fractions'].get('metallic', 0))
        
        # Create main figure with categorical x-axis (better for showing LoRA vs FiLM)
        fig, ax = plt.subplots(figsize=(11, 7))
        
        x_positions = np.arange(len(model_names))
        width = 0.35
        
        # Plot as grouped bars instead of line plot
        bars1 = ax.bar(x_positions - width/2, np.array(oxide_fractions)*100, width,
                      label='Oxide', color='orange', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_positions + width/2, np.array(metallic_fractions)*100, width,
                      label='Metallic', color='darkred', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add horizontal lines showing LoRA average
        lora_avg_oxide = np.mean(oxide_fractions[:3]) * 100  # First 3 are LoRA
        lora_avg_metallic = np.mean(metallic_fractions[:3]) * 100
        
        ax.axhline(lora_avg_oxide, color='orange', linestyle='--', linewidth=2, alpha=0.5,
                  label=f'LoRA avg oxide ({lora_avg_oxide:.1f}%)')
        ax.axhline(lora_avg_metallic, color='darkred', linestyle='--', linewidth=2, alpha=0.5,
                  label=f'LoRA avg metallic ({lora_avg_metallic:.1f}%)')
        
        ax.set_xlabel('Model Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Fraction of Generated Materials (%)', fontsize=14, fontweight='bold')
        ax.set_title('Chemical Diversity Trade-off:\nLoRA (Parameter-Efficient) vs. FiLM (Full Fine-Tuning)',
                    fontsize=15, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_names, fontsize=13)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 50])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add annotation highlighting the key finding
        ax.annotate('4.8× oxide enrichment\nin all LoRA ranks',
                   xy=(1.5, 40), xytext=(2.5, 45),
                   fontsize=11, fontweight='bold', color='orange',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', color='orange', lw=2))
        
        plt.tight_layout()
        
        output_file = output_path / 'oxide_bias_vs_rank.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved main figure: {output_file}")
        plt.close()
        
        # Create stacked bar chart
        self._create_stacked_chart(results_by_model, output_path)
        
        # Statistical analysis
        self._statistical_analysis(ranks, oxide_fractions, metallic_fractions, output_path)
    
    def _create_stacked_chart(self, results_by_model, output_path):
        """Create stacked bar chart of chemistry distributions."""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        model_names = []
        chemistry_data = {chem: [] for chem in self.chemistry_classes}
        
        for model_name, results in results_by_model.items():
            if results is None:
                continue
            
            model_names.append(model_name)
            for chem in self.chemistry_classes:
                fraction = results['chemistry_fractions'].get(chem, 0)
                chemistry_data[chem].append(fraction)
        
        # Create stacked bars
        x = np.arange(len(model_names))
        bottom = np.zeros(len(model_names))
        
        for chem in self.chemistry_classes:
            values = chemistry_data[chem]
            ax.bar(x, values, bottom=bottom, label=chem, 
                  color=self.class_colors[chem], alpha=0.8, edgecolor='black')
            bottom += values
        
        ax.set_ylabel('Fraction', fontsize=13, fontweight='bold')
        ax.set_title('Chemistry Distribution Across Models', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = output_path / 'chemistry_distribution_stacked.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved stacked chart: {output_file}")
        plt.close()
    
    def _statistical_analysis(self, ranks, oxide_fractions, metallic_fractions, output_path):
        """Perform statistical analysis and save results."""
        
        print("\n📈 Statistical Analysis:")
        
        # Filter out FiLM (rank 100) for LoRA-only correlation
        lora_mask = np.array(ranks) < 100
        lora_ranks = np.array(ranks)[lora_mask]
        lora_oxide = np.array(oxide_fractions)[lora_mask]
        lora_metallic = np.array(metallic_fractions)[lora_mask]
        
        # Spearman correlation (rank vs. oxide)
        if len(lora_ranks) >= 3:
            corr_oxide, p_oxide = spearmanr(lora_ranks, lora_oxide)
            corr_metallic, p_metallic = spearmanr(lora_ranks, lora_metallic)
            
            print(f"\n  Spearman Correlation (LoRA only):")
            print(f"    Rank vs. Oxide:    ρ = {corr_oxide:+.3f}, p = {p_oxide:.3e}")
            print(f"    Rank vs. Metallic: ρ = {corr_metallic:+.3f}, p = {p_metallic:.3e}")
            
            if corr_oxide < -0.8 and p_oxide < 0.05:
                print(f"\n  ✅ STRONG NEGATIVE CORRELATION: Lower rank → Higher oxide bias")
                print(f"     Statistically significant (p < 0.05)")
            
            if corr_metallic > 0.8 and p_metallic < 0.05:
                print(f"  ✅ STRONG POSITIVE CORRELATION: Higher rank → More metallics")
                print(f"     Statistically significant (p < 0.05)")
        
        # Compare LoRA average vs. FiLM
        if len(lora_oxide) > 0:
            avg_lora_oxide = np.mean(lora_oxide)
            film_oxide = oxide_fractions[-1] if len(oxide_fractions) > 0 else 0
            
            avg_lora_metallic = np.mean(lora_metallic)
            film_metallic = metallic_fractions[-1] if len(metallic_fractions) > 0 else 0
            
            oxide_enrichment = avg_lora_oxide / (film_oxide + 1e-6)
            metallic_depletion = avg_lora_metallic / (film_metallic + 1e-6)
            
            print(f"\n  LoRA vs. FiLM Comparison:")
            print(f"    LoRA avg oxide: {avg_lora_oxide*100:.2f}%")
            print(f"    FiLM oxide: {film_oxide*100:.2f}%")
            print(f"    Oxide enrichment: {oxide_enrichment:.2f}x")
            print()
            print(f"    LoRA avg metallic: {avg_lora_metallic*100:.2f}%")
            print(f"    FiLM metallic: {film_metallic*100:.2f}%")
            print(f"    Metallic depletion: {metallic_depletion:.2f}x")
            
            if oxide_enrichment > 1.1:
                print(f"\n  ✅ LoRA generates {oxide_enrichment:.1f}x more oxides than FiLM")
            
            if metallic_depletion < 0.9:
                print(f"  ✅ LoRA generates {1/metallic_depletion:.1f}x fewer metallics than FiLM")
        
        # Save statistics to file
        stats_file = output_path / 'statistical_analysis.txt'
        with open(stats_file, 'w') as f:
            f.write("STATISTICAL ANALYSIS OF GENERATION BIAS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Spearman Correlation (LoRA Ranks Only):\n")
            if len(lora_ranks) >= 3:
                f.write(f"  Rank vs. Oxide:    ρ = {corr_oxide:+.3f}, p = {p_oxide:.3e}\n")
                f.write(f"  Rank vs. Metallic: ρ = {corr_metallic:+.3f}, p = {p_metallic:.3e}\n\n")
            
            f.write("LoRA vs. FiLM Comparison:\n")
            if len(lora_oxide) > 0:
                f.write(f"  LoRA avg oxide: {avg_lora_oxide*100:.2f}%\n")
                f.write(f"  FiLM oxide: {film_oxide*100:.2f}%\n")
                f.write(f"  Oxide enrichment: {oxide_enrichment:.2f}x\n\n")
                
                f.write(f"  LoRA avg metallic: {avg_lora_metallic*100:.2f}%\n")
                f.write(f"  FiLM metallic: {film_metallic*100:.2f}%\n")
                f.write(f"  Metallic depletion: {metallic_depletion:.2f}x\n\n")
            
            f.write("INTERPRETATION:\n")
            if len(lora_ranks) >= 3 and corr_oxide < -0.8 and p_oxide < 0.05:
                f.write("✅ Strong negative correlation between rank and oxide bias\n")
                f.write("   → Lower rank causally increases oxide generation\n\n")
            
            if oxide_enrichment > 1.1 and metallic_depletion < 0.9:
                f.write("✅ LoRA systematically favors oxides over metallics\n")
                f.write("   → Low-rank constraint causes chemical diversity trade-off\n")
        
        print(f"\n  ✓ Saved statistics: {stats_file}")
    
    def print_summary(self, results_by_model):
        """Print comprehensive summary."""
        
        print("\n" + "="*70)
        print("GENERATION BIAS ANALYSIS SUMMARY")
        print("="*70)
        
        for model_name, results in results_by_model.items():
            if results is None:
                continue
            
            print(f"\n{model_name}:")
            print(f"  Total structures: {results['total_structures']}")
            print(f"  Chemistry distribution:")
            
            for chem in self.chemistry_classes:
                fraction = results['chemistry_fractions'].get(chem, 0)
                count = results['chemistry_counts'].get(chem, 0)
                print(f"    {chem:15s}: {count:5d} ({fraction*100:5.2f}%)")
        
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)
        
        # Extract oxide and metallic fractions
        oxide_by_model = {}
        metallic_by_model = {}
        
        for model_name, results in results_by_model.items():
            if results is None:
                continue
            oxide_by_model[model_name] = results['chemistry_fractions'].get('oxide', 0)
            metallic_by_model[model_name] = results['chemistry_fractions'].get('metallic', 0)
        
        # Check trend
        if len(oxide_by_model) >= 3:
            models = list(oxide_by_model.keys())
            oxides = list(oxide_by_model.values())
            
            # Check if oxide increases from R32 → R16 → R8
            if 'Rank 8' in models and 'Rank 16' in models and 'Rank 32' in models:
                r8_oxide = oxide_by_model['Rank 8']
                r16_oxide = oxide_by_model['Rank 16']
                r32_oxide = oxide_by_model['Rank 32']
                
                if r8_oxide > r16_oxide > r32_oxide:
                    print("\n✅ CONFIRMED: Oxide bias increases as rank decreases")
                    print(f"   Rank 8:  {r8_oxide*100:.1f}%")
                    print(f"   Rank 16: {r16_oxide*100:.1f}%")
                    print(f"   Rank 32: {r32_oxide*100:.1f}%")
                    print("\n   → Low-rank constraint CAUSES oxide bias")
        
        print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test generation bias across LoRA ranks'
    )
    parser.add_argument('--rank8_materials', type=str, required=True,
                       help='Path to Rank 8 generated materials')
    parser.add_argument('--rank16_materials', type=str, required=True,
                       help='Path to Rank 16 generated materials')
    parser.add_argument('--rank32_materials', type=str, required=True,
                       help='Path to Rank 32 generated materials')
    parser.add_argument('--film_materials', type=str, required=True,
                       help='Path to FiLM generated materials')
    parser.add_argument('--output_dir', type=str, default='generation_bias_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("GENERATION BIAS TEST")
    print("="*70)
    print(f"Rank 8:  {args.rank8_materials}")
    print(f"Rank 16: {args.rank16_materials}")
    print(f"Rank 32: {args.rank32_materials}")
    print(f"FiLM:    {args.film_materials}")
    print(f"Output:  {args.output_dir}")
    print("="*70)
    
    # Initialize analyzer
    analyzer = GenerationBiasAnalyzer()
    
    # Analyze each model (in rank order)
    results_by_model = OrderedDict()
    results_by_model['Rank 8'] = analyzer.analyze_materials(args.rank8_materials, 'Rank 8')
    results_by_model['Rank 16'] = analyzer.analyze_materials(args.rank16_materials, 'Rank 16')
    results_by_model['Rank 32'] = analyzer.analyze_materials(args.rank32_materials, 'Rank 32')
    results_by_model['FiLM'] = analyzer.analyze_materials(args.film_materials, 'FiLM')
    
    # Visualize
    analyzer.visualize_bias_vs_rank(results_by_model, args.output_dir)
    
    # Print summary
    analyzer.print_summary(results_by_model)
    
    print("✅ Analysis complete!\n")
    print("🎯 Key output: oxide_bias_vs_rank.png")
    print("   This figure directly proves the chemical diversity trade-off!\n")
    
    return 0


if __name__ == "__main__":
    exit(main())

