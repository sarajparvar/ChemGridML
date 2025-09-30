#!/usr/bin/env python3
"""
Minimal Feature Analysis - Using existing model results
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sqlite3

def analyze_existing_model_performance(db_path, dataset_name="CYCLODEXTRIN"):
    """Analyze performance using existing predictions from database"""
    
    print("="*60)
    print("ANALYZING EXISTING MODEL PERFORMANCE")
    print("="*60)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Get all XGBoost results
    query = '''
        SELECT p.fingerprint, p.model_name, p.prediction, dt.target_value, p.seed
        FROM predictions p
        JOIN dataset_targets dt ON p.dataset_name = dt.dataset_name AND p.data_index = dt.data_index
        WHERE p.dataset_name = ? AND p.model_name = 'XGBoost'
    '''
    
    df = pd.read_sql_query(query, conn, params=[dataset_name])
    conn.close()
    
    if df.empty:
        print("No XGBoost results found in database!")
        return
    
    print(f"Found {len(df)} XGBoost predictions")
    print(f"Fingerprint types: {df['fingerprint'].unique()}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    
    # Analyze performance by fingerprint type
    results = {}
    
    for fingerprint in df['fingerprint'].unique():
        fp_data = df[df['fingerprint'] == fingerprint]
        
        # Calculate metrics for each seed
        seed_results = []
        for seed in fp_data['seed'].unique():
            seed_data = fp_data[fp_data['seed'] == seed]
            
            y_true = seed_data['target_value'].values
            y_pred_proba = seed_data['prediction'].values
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred_proba)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            seed_results.append({
                'seed': seed,
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_samples': len(seed_data)
            })
        
        # Calculate summary statistics
        metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1']
        summary = {}
        
        for metric in metrics:
            values = [r[metric] for r in seed_results]
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_values'] = values
        
        results[fingerprint] = {
            'summary': summary,
            'seed_results': seed_results,
            'n_total_samples': len(fp_data)
        }
    
    # Print results
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY BY FINGERPRINT TYPE")
    print("="*80)
    
    best_fingerprint = None
    best_auc = -1
    
    for fingerprint, data in results.items():
        summary = data['summary']
        
        print(f"\n{fingerprint} Fingerprint:")
        print(f"  AUC:       {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
        print(f"  Accuracy:  {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        print(f"  Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
        print(f"  Recall:    {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
        print(f"  F1-Score:  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        print(f"  Samples:   {data['n_total_samples']}")
        
        if summary['auc_mean'] > best_auc:
            best_auc = summary['auc_mean']
            best_fingerprint = fingerprint
    
    print(f"\n🏆 Best performing fingerprint: {best_fingerprint} (AUC: {best_auc:.4f})")
    
    # Create visualization
    output_dir = f"./model_analysis_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'XGBoost Performance Comparison - {dataset_name}', fontsize=16, fontweight='bold')
    
    fingerprints = list(results.keys())
    metrics = ['auc', 'accuracy', 'precision', 'recall']
    metric_labels = ['AUC', 'Accuracy', 'Precision', 'Recall']
    
    # Use colorblind-safe colors
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i//2, i%2]
        
        means = [results[fp]['summary'][f'{metric}_mean'] for fp in fingerprints]
        stds = [results[fp]['summary'][f'{metric}_std'] for fp in fingerprints]
        
        bars = ax.bar(fingerprints, means, yerr=stds, capsize=5, 
                     color=colors[:len(fingerprints)], alpha=0.8)
        
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} by Fingerprint Type', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Clean up
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results to CSV
    detailed_results = []
    for fingerprint, data in results.items():
        for seed_result in data['seed_results']:
            row = {'fingerprint': fingerprint, **seed_result}
            detailed_results.append(row)
    
    detailed_df = pd.DataFrame(detailed_results)
    csv_path = os.path.join(output_dir, 'detailed_results.csv')
    detailed_df.to_csv(csv_path, index=False)
    
    # Save summary
    summary_data = []
    for fingerprint, data in results.items():
        summary = data['summary']
        row = {'fingerprint': fingerprint, **summary, 'n_samples': data['n_total_samples']}
        # Remove the values lists for CSV
        row = {k: v for k, v in row.items() if not k.endswith('_values')}
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'summary_results.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nGenerated files in '{output_dir}':")
    print("  - performance_comparison.png")
    print("  - detailed_results.csv")
    print("  - summary_results.csv")
    
    return results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python model_analysis.py <master_job_id>")
        sys.exit(1)
    
    master_job_id = sys.argv[1]
    
    # Find database path
    base = Path('./studies')
    path = base / master_job_id
    directories = [item.name for item in path.iterdir() if item.is_dir()]
    
    if not directories:
        print(f"No directories found in {path}")
        sys.exit(1)
    
    dataset_name = directories[0]
    db_path = path / dataset_name / "predictions.db"
    
    print(f"Analyzing dataset: {dataset_name}")
    print(f"Database path: {db_path}")
    
    try:
        results = analyze_existing_model_performance(str(db_path), dataset_name)
        print(f"\nAnalysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
