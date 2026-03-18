#!/usr/bin/env python3
"""
Calculate efficiency metrics for LoRA vs FILM adapters
This script extracts actual parameter counts, model sizes, and other efficiency metrics
"""

import os
import sys
import json
import torch
import yaml
from pathlib import Path
import time
import psutil
import subprocess

# Add MatterGen to path
sys.path.append('/home/arash/Projects/mattergen-main')

def get_file_size_mb(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0

def count_parameters_in_checkpoint(checkpoint_path):
    """Count parameters in a PyTorch checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Count total parameters
        total_params = 0
        trainable_params = 0
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                param_count = param.numel()
                total_params += param_count
                
                # Check if parameter is trainable (not frozen)
                if 'lora' in name.lower() or 'adapter' in name.lower():
                    trainable_params += param_count
                elif 'film' in name.lower() or 'full_finetuning' in name.lower():
                    trainable_params += param_count
        
        return total_params, trainable_params
    except Exception as e:
        print(f"Error counting parameters in {checkpoint_path}: {e}")
        return 0, 0

def analyze_lora_efficiency():
    """Analyze LoRA adapter efficiency"""
    print("🔍 ANALYZING LoRA ADAPTER EFFICIENCY")
    print("=" * 50)
    
    # Find LoRA checkpoints
    lora_checkpoints = []
    for root, dirs, files in os.walk('/home/arash/Projects/mattergen-main/outputs'):
        for file in files:
            if file.endswith('.ckpt') and 'lora' in file.lower():
                lora_checkpoints.append(os.path.join(root, file))
    
    if not lora_checkpoints:
        print("❌ No LoRA checkpoints found")
        return None
    
    # Use the most recent checkpoint
    latest_checkpoint = max(lora_checkpoints, key=os.path.getctime)
    print(f"📁 Using checkpoint: {latest_checkpoint}")
    
    # Get file size
    file_size_mb = get_file_size_mb(latest_checkpoint)
    print(f"📊 File size: {file_size_mb:.1f} MB")
    
    # Count parameters
    total_params, trainable_params = count_parameters_in_checkpoint(latest_checkpoint)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    # Calculate efficiency metrics
    if total_params > 0:
        trainable_ratio = trainable_params / total_params
        print(f"📊 Trainable ratio: {trainable_ratio:.1%}")
    
    return {
        'file_size_mb': file_size_mb,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        'checkpoint_path': latest_checkpoint
    }

def analyze_film_efficiency():
    """Analyze FILM adapter efficiency"""
    print("\n🔍 ANALYZING FILM ADAPTER EFFICIENCY")
    print("=" * 50)
    
    # Use specific FILM checkpoint
    film_checkpoint = '/home/arash/Projects/mattergen-main/npj_submission_package/models/film_adapter/checkpoints/last.ckpt'
    
    if not os.path.exists(film_checkpoint):
        print("❌ FILM checkpoint not found")
        return None
    
    print(f"📁 Using checkpoint: {film_checkpoint}")
    
    # Get file size
    file_size_mb = get_file_size_mb(film_checkpoint)
    print(f"📊 File size: {file_size_mb:.1f} MB")
    
    # Count parameters
    total_params, trainable_params = count_parameters_in_checkpoint(film_checkpoint)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    # FILM is full fine-tuning, so all parameters are trainable
    if trainable_params == 0 and total_params > 0:
        trainable_params = total_params
        print("📊 Note: FILM uses full fine-tuning (all parameters trainable)")
    
    # Calculate efficiency metrics
    if total_params > 0:
        trainable_ratio = trainable_params / total_params
        print(f"📊 Trainable ratio: {trainable_ratio:.1%}")
    
    return {
        'file_size_mb': file_size_mb,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        'checkpoint_path': film_checkpoint
    }

def analyze_base_model():
    """Analyze base model for comparison"""
    print("\n🔍 ANALYZING BASE MODEL")
    print("=" * 50)
    
    base_model_path = '/home/arash/Projects/mattergen-main/checkpoints/mp_20_base'
    
    if not os.path.exists(base_model_path):
        print("❌ Base model not found")
        return None
    
    # Find base model files
    base_files = []
    for root, dirs, files in os.walk(base_model_path):
        for file in files:
            if file.endswith(('.ckpt', '.pth', '.bin')):
                base_files.append(os.path.join(root, file))
    
    if not base_files:
        print("❌ No base model files found")
        return None
    
    # Use the largest file (likely the main model)
    largest_file = max(base_files, key=os.path.getsize)
    print(f"📁 Using base model: {largest_file}")
    
    # Get file size
    file_size_mb = get_file_size_mb(largest_file)
    print(f"📊 File size: {file_size_mb:.1f} MB")
    
    # Count parameters
    total_params, trainable_params = count_parameters_in_checkpoint(largest_file)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    return {
        'file_size_mb': file_size_mb,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        'model_path': largest_file
    }

def calculate_efficiency_ratios(lora_data, film_data, base_data):
    """Calculate efficiency ratios"""
    print("\n📊 EFFICIENCY COMPARISON")
    print("=" * 50)
    
    if not all([lora_data, film_data, base_data]):
        print("❌ Missing data for comparison")
        return None
    
    # Parameter efficiency
    lora_efficiency = lora_data['trainable_params'] / base_data['total_params']
    film_efficiency = film_data['trainable_params'] / base_data['total_params']
    
    print(f"🎯 PARAMETER EFFICIENCY:")
    print(f"  • LoRA: {lora_efficiency:.1%} of base model parameters")
    print(f"  • FILM: {film_efficiency:.1%} of base model parameters")
    print(f"  • LoRA advantage: {film_efficiency/lora_efficiency:.1f}x more efficient")
    
    # Size efficiency
    lora_size_ratio = lora_data['file_size_mb'] / base_data['file_size_mb']
    film_size_ratio = film_data['file_size_mb'] / base_data['file_size_mb']
    
    print(f"\n🎯 SIZE EFFICIENCY:")
    print(f"  • LoRA: {lora_size_ratio:.1%} of base model size")
    print(f"  • FILM: {film_size_ratio:.1%} of base model size")
    print(f"  • LoRA advantage: {film_size_ratio/lora_size_ratio:.1f}x smaller")
    
    return {
        'lora_efficiency': lora_efficiency,
        'film_efficiency': film_efficiency,
        'lora_size_ratio': lora_size_ratio,
        'film_size_ratio': film_size_ratio,
        'parameter_advantage': film_efficiency/lora_efficiency,
        'size_advantage': film_size_ratio/lora_size_ratio
    }

def main():
    """Main function"""
    print("🚀 CALCULATING EFFICIENCY METRICS")
    print("=" * 60)
    
    # Analyze each adapter type
    lora_data = analyze_lora_efficiency()
    film_data = analyze_film_efficiency()
    base_data = analyze_base_model()
    
    # Calculate efficiency ratios
    efficiency_ratios = calculate_efficiency_ratios(lora_data, film_data, base_data)
    
    # Save results
    results = {
        'lora': lora_data,
        'film': film_data,
        'base': base_data,
        'efficiency_ratios': efficiency_ratios,
        'timestamp': time.time()
    }
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save to JSON
    with open('results/efficiency_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to results/efficiency_metrics.json")
    
    # Print summary table
    if efficiency_ratios:
        print("\n📊 SUMMARY TABLE")
        print("=" * 60)
        print("| Metric | LoRA | FILM | LoRA Advantage |")
        print("|--------|------|------|----------------|")
        print(f"| Trainable Params | {lora_data['trainable_params']:,} | {film_data['trainable_params']:,} | {efficiency_ratios['parameter_advantage']:.1f}x |")
        print(f"| File Size (MB) | {lora_data['file_size_mb']:.1f} | {film_data['file_size_mb']:.1f} | {efficiency_ratios['size_advantage']:.1f}x |")
        print(f"| Efficiency | {lora_data['trainable_ratio']:.1%} | {film_data['trainable_ratio']:.1%} | - |")

if __name__ == "__main__":
    main()
