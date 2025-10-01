# benchmark_manager.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error
from typing import Dict, List, Tuple, Optional
import os, sys
from pathlib import Path
from datetime import datetime
from database_manager import DatabaseManager
from scipy.stats import friedmanchisquare, ttest_rel
from scipy.stats import rankdata
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class BenchmarkManager:
    """Analyzer for molecular property prediction results with multiple train-test splits"""
    
    def __init__(self, db_manager, save_dir: str = "analysis_results"):
        self.db_manager = db_manager
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Store computed results
        self.results = {}
        self.statistical_results = {}
    
    def is_classification_dataset(self, dataset_name: str) -> bool:
        """Determine if dataset is classification by checking if all targets are 0 or 1"""
        with self.db_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT target_value FROM dataset_targets 
                WHERE dataset_name = ?
            ''', (dataset_name,))
            
            unique_targets = [row[0] for row in cursor.fetchall()]
            
            # Check if all values are either 0 or 1
            return all(target in [0.0, 1.0] for target in unique_targets)
    
    def compute_auroc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute AUROC score for binary classification"""
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError as e:
            print(f"Warning: Could not compute AUROC - {e}")
            return np.nan
    
    def compute_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Square Error for regression"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def get_test_predictions(self, dataset_name: str) -> pd.DataFrame:
        """Get all test predictions for a dataset"""
        df = self.db_manager.get_predictions_dataframe(dataset_name)
        print(df)
        return df[df['split_type'] == 'random']
    
    def compute_metrics_for_dataset(self, dataset_name: str) -> Dict:
        """Compute metrics for all fingerprint/model/seed combinations in a dataset"""
        print(f"Processing dataset: {dataset_name}")
        
        # Get test predictions
        test_df = self.get_test_predictions(dataset_name)
        
        if test_df.empty:
            print(f"No test predictions found for dataset {dataset_name}")
            return {}
        
        # Determine if classification or regression
        is_classification = self.is_classification_dataset(dataset_name)
        metric_name = "AUROC" if is_classification else "RMSE"
        
        print(f"Dataset {dataset_name} identified as {'classification' if is_classification else 'regression'}")
        
        results = {
            'dataset': dataset_name,
            'metric': metric_name,
            'is_classification': is_classification,
            'scores': []
        }
        
        # Group by fingerprint, model, and seed
        groups = test_df.groupby(['fingerprint', 'model_name', 'seed'])
        
        for (fingerprint, model, seed), group in groups:
            y_true = group['target_value'].values
            y_pred = group['prediction'].values
            
            if is_classification:
                score = self.compute_auroc(y_true, y_pred)
            else:
                score = self.compute_rmse(y_true, y_pred)
            
            results['scores'].append({
                'fingerprint': fingerprint,
                'model': model,
                'seed': seed,
                'score': score,
                'n_samples': len(y_true)
            })
        
        print(f"Computed {len(results['scores'])} metric scores for {dataset_name}")
        print(results)
        return results
    
    def analyze_all_datasets(self) -> Dict:
        """Analyze all datasets in the database"""
        # Get all unique datasets
        with self.db_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT dataset_name FROM dataset_targets')
            datasets = [row[0] for row in cursor.fetchall()]
        
        print(f"Found {len(datasets)} datasets: {datasets}")
        
        # Analyze each dataset
        for dataset in datasets:
            self.results[dataset] = self.compute_metrics_for_dataset(dataset)
        
        return self.results
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics across all seeds for each fingerprint/model/dataset combination"""
        summary_data = []
        
        for dataset_name, dataset_results in self.results.items():
            if not dataset_results or not dataset_results['scores']:
                continue
                
            # Convert to DataFrame for easier manipulation
            scores_df = pd.DataFrame(dataset_results['scores'])
            
            # Group by fingerprint and model, compute statistics across seeds
            group_stats = scores_df.groupby(['fingerprint', 'model'])['score'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).reset_index()
            
            # Add dataset and metric information
            group_stats['dataset'] = dataset_name
            group_stats['metric'] = dataset_results['metric']
            group_stats['is_classification'] = dataset_results['is_classification']
            
            summary_data.append(group_stats)
        
        if summary_data:
            return pd.concat(summary_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_detailed_scores(self) -> pd.DataFrame:
        """Get individual seed scores for statistical analysis"""
        detailed_data = []
        
        for dataset_name, dataset_results in self.results.items():
            if not dataset_results or not dataset_results['scores']:
                continue
                
            scores_df = pd.DataFrame(dataset_results['scores'])
            scores_df['dataset'] = dataset_name
            scores_df['metric'] = dataset_results['metric']
            scores_df['is_classification'] = dataset_results['is_classification']
            scores_df['method'] = scores_df['fingerprint'] + '_' + scores_df['model']
            
            detailed_data.append(scores_df)
        
        if detailed_data:
            return pd.concat(detailed_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def friedman_ranking_analysis(self) -> Dict:
        """Perform Friedman test and ranking analysis"""
        print("\n=== Friedman Ranking Analysis ===")
        
        detailed_df = self.get_detailed_scores()
        if detailed_df.empty:
            return {}
        
        # Separate classification and regression datasets
        classification_df = detailed_df[detailed_df['is_classification'] == True]
        regression_df = detailed_df[detailed_df['is_classification'] == False]
        
        results = {}
        
        for task_type, df in [('Classification', classification_df), ('Regression', regression_df)]:
            if df.empty:
                continue
                
            print(f"\n{task_type} Datasets:")
            
            # Create ranking matrix: rows=datasets, columns=methods
            datasets = df['dataset'].unique()
            
            # Find common methods across all datasets for this task type
            common_methods = set(df['method'].unique())
            for dataset in datasets:
                dataset_methods = set(df[df['dataset'] == dataset]['method'].unique())
                common_methods = common_methods.intersection(dataset_methods)
            
            common_methods = sorted(list(common_methods))
            
            if len(common_methods) < 2:
                print(f"Insufficient common methods ({len(common_methods)}) for Friedman test")
                results[task_type] = {'error': 'Insufficient common methods'}
                continue
            
            print(f"Found {len(common_methods)} common methods across {len(datasets)} datasets")
            
            rank_matrix = []
            for dataset in datasets:
                dataset_data = df[df['dataset'] == dataset]
                method_means = dataset_data.groupby('method')['score'].mean()
                
                # Only use common methods
                method_means = method_means[common_methods]
                
                # Check if we have all methods for this dataset
                if len(method_means) != len(common_methods):
                    print(f"Warning: Dataset {dataset} missing some methods, skipping")
                    continue
                
                # Rank methods (1=best)
                if task_type == 'Classification':  # Higher AUROC is better
                    ranks = rankdata(-method_means.values, method='average')
                else:  # Lower RMSE is better
                    ranks = rankdata(method_means.values, method='average')
                
                rank_matrix.append(ranks)
            
            if len(rank_matrix) < 2:
                print(f"Insufficient datasets ({len(rank_matrix)}) for Friedman test")
                results[task_type] = {'error': 'Insufficient datasets'}
                continue
            
            rank_matrix = np.array(rank_matrix)
            
            # Friedman test
            try:
                statistic, p_value = friedmanchisquare(*rank_matrix.T)
                
                # Calculate mean ranks
                mean_ranks = np.mean(rank_matrix, axis=0)
                method_rankings = list(zip(common_methods, mean_ranks))
                method_rankings.sort(key=lambda x: x[1])  # Sort by rank (lower is better)
                
                results[task_type] = {
                    'friedman_stat': statistic,
                    'friedman_p': p_value,
                    'rankings': method_rankings,
                    'significant': p_value < 0.05,
                    'n_datasets': len(rank_matrix),
                    'n_methods': len(common_methods)
                }
                
                print(f"Friedman test: χ² = {statistic:.3f}, p = {p_value:.4f}")
                print(f"Based on {len(rank_matrix)} datasets and {len(common_methods)} methods")
                if p_value < 0.05:
                    print("Significant differences between methods detected!")
                    print("Top 5 methods:")
                    for i, (method, rank) in enumerate(method_rankings[:5]):
                        fp, model = method.split('_')
                        print(f"  {i+1}. {fp} + {model} (avg rank: {rank:.2f})")
                else:
                    print("No significant differences between methods")
                    
            except Exception as e:
                print(f"Error in Friedman test: {e}")
                results[task_type] = {'error': str(e)}
        
        return results
    
    def component_anova_analysis(self) -> Dict:
        """Perform ANOVA analysis on fingerprints and models"""
        print("\n=== Component ANOVA Analysis ===")
        
        detailed_df = self.get_detailed_scores()
        if detailed_df.empty:
            return {}
        
        # Separate classification and regression
        classification_df = detailed_df[detailed_df['is_classification'] == True]
        regression_df = detailed_df[detailed_df['is_classification'] == False]
        
        results = {}
        
        for task_type, df in [('Classification', classification_df), ('Regression', regression_df)]:
            if df.empty:
                continue
                
            print(f"\n{task_type} Datasets:")
            
            try:
                # Main effects model
                model_main = ols('score ~ C(fingerprint) + C(model) + C(dataset)', data=df).fit()
                anova_main = anova_lm(model_main)
                
                # Interaction model  
                model_int = ols('score ~ C(fingerprint) * C(model) + C(dataset)', data=df).fit()
                anova_int = anova_lm(model_int)
                
                # Component means
                fp_means = df.groupby('fingerprint')['score'].mean().sort_values(ascending=(task_type=='Regression'))
                model_means = df.groupby('model')['score'].mean().sort_values(ascending=(task_type=='Regression'))
                
                results[task_type] = {
                    'anova_main': anova_main,
                    'anova_interaction': anova_int,
                    'fingerprint_means': fp_means,
                    'model_means': model_means
                }
                
                # Print results
                fp_p = anova_main.loc['C(fingerprint)', 'PR(>F)']
                model_p = anova_main.loc['C(model)', 'PR(>F)']
                int_p = anova_int.loc['C(fingerprint):C(model)', 'PR(>F)']
                
                print(f"Fingerprint effect: p = {fp_p:.4f} {'***' if fp_p < 0.001 else '**' if fp_p < 0.01 else '*' if fp_p < 0.05 else ''}")
                print(f"Model effect: p = {model_p:.4f} {'***' if model_p < 0.001 else '**' if model_p < 0.01 else '*' if model_p < 0.05 else ''}")
                print(f"Interaction effect: p = {int_p:.4f} {'***' if int_p < 0.001 else '**' if int_p < 0.01 else '*' if int_p < 0.05 else ''}")
                
                if fp_p < 0.05:
                    print("Best fingerprints:")
                    for i, (fp, score) in enumerate(fp_means.items()):
                        if i < 3:
                            print(f"  {i+1}. {fp}: {score:.4f}")
                
                if model_p < 0.05:
                    print("Best models:")
                    for i, (model, score) in enumerate(model_means.items()):
                        if i < 3:
                            print(f"  {i+1}. {model}: {score:.4f}")
                            
            except Exception as e:
                print(f"Error in ANOVA: {e}")
                results[task_type] = {'error': str(e)}
        
        return results
    
    def dataset_winners_analysis(self) -> Dict:
        """Find statistical winners for each dataset"""
        print("\n=== Dataset Winners Analysis ===")
        
        summary_df = self.get_summary_statistics()
        detailed_df = self.get_detailed_scores()
        
        winners = {}
        
        for dataset in summary_df['dataset'].unique():
            dataset_summary = summary_df[summary_df['dataset'] == dataset]
            dataset_detailed = detailed_df[detailed_df['dataset'] == dataset]
            
            is_classification = dataset_summary['is_classification'].iloc[0]
            
            # Find best method
            if is_classification:
                best_idx = dataset_summary['mean'].idxmax()
            else:
                best_idx = dataset_summary['mean'].idxmin()
            
            best_method = dataset_summary.loc[best_idx]
            best_fp = best_method['fingerprint']
            best_model = best_method['model']
            best_score = best_method['mean']
            
            # Get scores for statistical testing
            best_scores = dataset_detailed[
                (dataset_detailed['fingerprint'] == best_fp) & 
                (dataset_detailed['model'] == best_model)
            ]['score'].values
            
            # Test against second best
            dataset_summary_sorted = dataset_summary.sort_values('mean', ascending=not is_classification)
            if len(dataset_summary_sorted) > 1:
                second_best = dataset_summary_sorted.iloc[1]
                second_fp = second_best['fingerprint']
                second_model = second_best['model']
                
                second_scores = dataset_detailed[
                    (dataset_detailed['fingerprint'] == second_fp) & 
                    (dataset_detailed['model'] == second_model)
                ]['score'].values
                
                if len(best_scores) == len(second_scores) and len(best_scores) > 1:
                    _, p_value = ttest_rel(best_scores, second_scores)
                    significant = p_value < 0.05
                else:
                    p_value = np.nan
                    significant = False
            else:
                p_value = np.nan
                significant = False
            
            winners[dataset] = {
                'winner': f"{best_fp}+{best_model}",
                'score': best_score,
                'p_value': p_value,
                'significant': significant
            }
            
            sig_text = " (significant)" if significant else ""
            print(f"{dataset}: {best_fp}+{best_model} = {best_score:.4f} {p_value:.2f} {sig_text}")
        
        return winners
    
    def run_statistical_analysis(self) -> Dict:
        """Run complete statistical analysis"""
        if not self.results:
            print("No results available. Run analyze_all_datasets() first.")
            return {}
        
        print("Running Statistical Analysis...")
        
        # Run all analyses
        self.statistical_results = {
            'dataset_winners': self.dataset_winners_analysis(),
            'friedman_ranking': self.friedman_ranking_analysis(), 
            'component_anova': self.component_anova_analysis()
        }
        
        return self.statistical_results
    
    def compute_pairwise_significance(self, dataset_name: str, detailed_df: pd.DataFrame):
        """Compute pairwise statistical significance for methods within a dataset"""
        from scipy.stats import ttest_rel, wilcoxon
        from itertools import combinations
        
        dataset_data = detailed_df[detailed_df['dataset'] == dataset_name]
        methods = dataset_data['method'].unique()
        
        significance_matrix = {}
        
        # Get scores for each method
        method_scores = {}
        for method in methods:
            scores = dataset_data[dataset_data['method'] == method]['score'].values
            method_scores[method] = scores
        
        # Compute pairwise comparisons
        for method1, method2 in combinations(methods, 2):
            scores1 = method_scores[method1]
            scores2 = method_scores[method2]
            
            if len(scores1) == len(scores2) and len(scores1) > 1:
                try:
                    # Use paired t-test since these are the same seeds
                    _, p_value = ttest_rel(scores1, scores2)
                    significance_matrix[(method1, method2)] = p_value
                    significance_matrix[(method2, method1)] = p_value
                except:
                    significance_matrix[(method1, method2)] = 1.0
                    significance_matrix[(method2, method1)] = 1.0
            else:
                significance_matrix[(method1, method2)] = 1.0
                significance_matrix[(method2, method1)] = 1.0
        
        return significance_matrix
    
    def get_significance_stars(self, p_value):
        """Convert p-value to significance stars"""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return ""
    
    def print_winners_summary(self):
        """Print concise summary of winners"""
        if not self.statistical_results:
            print("Run statistical analysis first!")
            return
            
        print("\n" + "="*50)
        print("WINNERS SUMMARY")
        print("="*50)
        
        # Overall winners from Friedman test
        friedman_results = self.statistical_results.get('friedman_ranking', {})
        for task_type in ['Classification', 'Regression']:
            if task_type in friedman_results and 'rankings' in friedman_results[task_type]:
                rankings = friedman_results[task_type]['rankings']
                if rankings:
                    winner = rankings[0][0]
                    fp, model = winner.split('_')
                    rank = rankings[0][1]
                    print(f"\nOverall {task_type} Winner: {fp} + {model} (avg rank: {rank:.2f})")
        
        # Best components from ANOVA
        anova_results = self.statistical_results.get('component_anova', {})
        for task_type in ['Classification', 'Regression']:
            if task_type in anova_results and 'fingerprint_means' in anova_results[task_type]:
                fp_means = anova_results[task_type]['fingerprint_means']
                model_means = anova_results[task_type]['model_means']
                
                if not fp_means.empty and not model_means.empty:
                    best_fp = fp_means.index[0]
                    best_model = model_means.index[0]
                    print(f"{task_type} - Best Fingerprint: {best_fp} ({fp_means.iloc[0]:.4f})")
                    print(f"{task_type} - Best Model: {best_model} ({model_means.iloc[0]:.4f})")
    
    def plot_model_comparison_with_significance(self, figsize=(12, 8)):
        """Create a focused comparison plot showing XGBoost superiority with paired t-test significance"""
        if not self.results:
            print("No results to plot yet. Please run analyze_all_datasets() first.")
            return
        
        # Get detailed scores for statistical testing
        detailed_df = self.get_detailed_scores()
        
        if detailed_df.empty:
            print("No valid results to plot.")
            return
        
        # Group by dataset
        datasets = detailed_df['dataset'].unique()
        
        fig, axes = plt.subplots(1, len(datasets), figsize=figsize)
        if len(datasets) == 1:
            axes = [axes]
        
        fig.suptitle('Model Performance Comparison: XGBoost vs Others\n(Paired t-test significance)', 
                     fontsize=16, fontweight='bold')
        
        # Define colors for models
        models = sorted(detailed_df['model'].unique())
        model_colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        model_color_map = {model: model_colors[i] for i, model in enumerate(models)}
        
        # Make XGBoost stand out
        if 'XGBoost' in model_color_map:
            model_color_map['XGBoost'] = 'gold'
        
        for idx, dataset in enumerate(datasets):
            ax = axes[idx] if len(datasets) > 1 else axes[0]
            
            dataset_data = detailed_df[detailed_df['dataset'] == dataset]
            is_classification = dataset_data['is_classification'].iloc[0]
            metric_name = dataset_data['metric'].iloc[0]
            
            # Calculate mean performance for each model across all fingerprints and seeds
            model_means = []
            model_stds = []
            model_names = []
            significance_stars = []
            
            # Get XGBoost scores for comparison (across all fingerprints)
            xgboost_scores = dataset_data[dataset_data['model'] == 'XGBoost']['score'].values
            
            for model in models:
                model_data = dataset_data[dataset_data['model'] == model]
                if len(model_data) > 0:
                    scores = model_data['score'].values
                    model_means.append(np.mean(scores))
                    model_stds.append(np.std(scores))
                    model_names.append(model)
                    
                    # Perform paired t-test against XGBoost
                    if model != 'XGBoost' and len(scores) == len(xgboost_scores) and len(scores) > 1:
                        try:
                            from scipy.stats import ttest_rel
                            _, p_value = ttest_rel(xgboost_scores, scores)
                            stars = self.get_significance_stars(p_value)
                            significance_stars.append(stars)
                        except:
                            significance_stars.append("")
                    elif model == 'XGBoost':
                        significance_stars.append("REF")  # Reference model
                    else:
                        significance_stars.append("")
            
            # Create bar plot
            x_pos = np.arange(len(model_names))
            colors = [model_color_map.get(model, 'gray') for model in model_names]
            
            bars = ax.bar(x_pos, model_means, yerr=model_stds, capsize=5,
                         color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add significance stars above bars
            for i, (bar, stars) in enumerate(zip(bars, significance_stars)):
                height = bar.get_height() + model_stds[i]
                if stars == "REF":
                    # Mark XGBoost as reference with crown
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                           '👑', ha='center', va='bottom', fontsize=14)
                elif stars:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                           stars, ha='center', va='bottom', fontsize=14, fontweight='bold', color='red')
            
            # Customize the plot
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel(f'{metric_name}', fontsize=12)
            ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Set appropriate y-limits
            if is_classification:
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(bottom=0)
        
        # Add legend explaining significance
        legend_text = 'Significance vs XGBoost:\n* p<0.05  ** p<0.01  *** p<0.001\n👑 = Reference (XGBoost)'
        fig.text(0.02, 0.02, legend_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'xgboost_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_xgboost_significance_analysis(self):
        """Print detailed significance analysis showing XGBoost superiority"""
        if not self.results:
            print("No results available. Please run analyze_all_datasets() first.")
            return
        
        print("\n" + "="*80)
        print("XGBoost SUPERIORITY ANALYSIS")
        print("="*80)
        
        detailed_df = self.get_detailed_scores()
        
        for dataset in sorted(self.results.keys()):
            dataset_data = detailed_df[detailed_df['dataset'] == dataset]
            if dataset_data.empty:
                continue
                
            is_classification = dataset_data['is_classification'].iloc[0]
            metric_name = dataset_data['metric'].iloc[0]
            
            print(f"\nDataset: {dataset} ({metric_name})")
            print("-" * 60)
            
            # Get XGBoost scores
            xgboost_data = dataset_data[dataset_data['model'] == 'XGBoost']
            if xgboost_data.empty:
                print("No XGBoost results found for this dataset")
                continue
                
            xgboost_scores = xgboost_data['score'].values
            xgboost_mean = np.mean(xgboost_scores)
            xgboost_std = np.std(xgboost_scores)
            
            print(f"XGBoost (Reference): {xgboost_mean:.4f} ± {xgboost_std:.3f}")
            print()
            
            # Compare against other models
            models = [m for m in dataset_data['model'].unique() if m != 'XGBoost']
            
            print(f"{'Model':<15} {'Score':<12} {'Difference':<12} {'P-value':<10} {'Significance'}")
            print("-" * 60)
            
            for model in sorted(models):
                model_data = dataset_data[dataset_data['model'] == model]
                if model_data.empty:
                    continue
                    
                model_scores = model_data['score'].values
                model_mean = np.mean(model_scores)
                model_std = np.std(model_scores)
                
                # Paired t-test
                if len(model_scores) == len(xgboost_scores) and len(model_scores) > 1:
                    try:
                        from scipy.stats import ttest_rel
                        _, p_value = ttest_rel(xgboost_scores, model_scores)
                        
                        # Calculate effect size (difference)
                        if is_classification:
                            difference = xgboost_mean - model_mean  # Higher is better
                        else:
                            difference = model_mean - xgboost_mean  # Lower is better for RMSE
                        
                        stars = self.get_significance_stars(p_value)
                        sig_text = stars if stars else "n.s."
                        
                        print(f"{model:<15} {model_mean:<8.4f} {difference:>+8.4f} {p_value:<10.3f} {sig_text}")
                        
                    except Exception as e:
                        print(f"{model:<15} {model_mean:<8.4f} {'N/A':<12} {'N/A':<10} {'Error'}")
                else:
                    print(f"{model:<15} {model_mean:<8.4f} {'N/A':<12} {'N/A':<10} {'N/A'}")

    def print_xgboost_comparison_summary(self):
        """Print summary of XGBoost vs second-best comparisons"""
        print("\n" + "="*60)
        print("XGBOOST vs SECOND-BEST COMPARISON")
        print("="*60)
        
        df = self.get_summary_statistics()
        detailed_df = self.get_detailed_scores()
        
        for dataset in sorted(self.results.keys()):
            dataset_results = self.results[dataset]
            if not dataset_results or not dataset_results['scores']:
                continue
                
            dataset_df = df[df['dataset'] == dataset]
            metric_name = dataset_results['metric']
            is_classification = dataset_results['is_classification']
            
            # Find XGBoost and second-best performances
            xgboost_performances = []
            all_performances = []
            
            for _, row in dataset_df.iterrows():
                performance_data = {
                    'fingerprint': row['fingerprint'],
                    'model': row['model'], 
                    'mean': row['mean'],
                    'std': row['std'],
                    'method': f"{row['fingerprint']}_{row['model']}"
                }
                all_performances.append(performance_data)
                
                if row['model'].upper() == 'XGBOOST':
                    xgboost_performances.append(performance_data)
            
            # Sort to find best and second best
            if is_classification:
                all_performances.sort(key=lambda x: x['mean'], reverse=True)
                xgboost_performances.sort(key=lambda x: x['mean'], reverse=True)
            else:
                all_performances.sort(key=lambda x: x['mean'])
                xgboost_performances.sort(key=lambda x: x['mean'])
            
            # Find best XGBoost and overall second best
            best_xgboost = xgboost_performances[0] if xgboost_performances else None
            second_best_overall = None
            
            # Find second best that's not the best XGBoost
            for perf in all_performances:
                if best_xgboost and perf['method'] != best_xgboost['method']:
                    second_best_overall = perf
                    break
            
            print(f"\nDataset: {dataset} ({metric_name})")
            print("-" * 40)
            
            if best_xgboost and second_best_overall:
                print(f"Best XGBoost: {best_xgboost['fingerprint']}+{best_xgboost['model']} = {best_xgboost['mean']:.4f} ± {best_xgboost['std']:.4f}")
                print(f"Second Best:  {second_best_overall['fingerprint']}+{second_best_overall['model']} = {second_best_overall['mean']:.4f} ± {second_best_overall['std']:.4f}")
                
                # Statistical test
                xgboost_data = detailed_df[
                    (detailed_df['dataset'] == dataset) & 
                    (detailed_df['method'] == best_xgboost['method'])
                ]
                second_best_data = detailed_df[
                    (detailed_df['dataset'] == dataset) & 
                    (detailed_df['method'] == second_best_overall['method'])
                ]
                
                if len(xgboost_data) > 1 and len(second_best_data) > 1:
                    xgboost_scores = xgboost_data['score'].values
                    second_scores = second_best_data['score'].values
                    
                    min_len = min(len(xgboost_scores), len(second_scores))
                    if min_len > 1:
                        try:
                            from scipy.stats import ttest_rel
                            _, p_value = ttest_rel(xgboost_scores[:min_len], second_scores[:min_len])
                            stars = self.get_significance_stars(p_value)
                            
                            improvement = ((best_xgboost['mean'] - second_best_overall['mean']) / 
                                         second_best_overall['mean'] * 100)
                            
                            print(f"Improvement: {improvement:+.2f}%")
                            print(f"P-value: {p_value:.4f} {stars if stars else '(not significant)'}")
                            
                        except Exception as e:
                            print(f"Statistical test failed: {e}")
            else:
                print("XGBoost or comparison data not available")

    def plot_xgboost_vs_second_best(self, figsize=(16, 10)):
        """Create comparison plot showing XGBoost vs second-best with significance stars on XGBoost"""
        if not self.results:
            print("No results to plot yet. Please run analyze_all_datasets() first.")
            return
        
        # Get summary statistics
        df = self.get_summary_statistics()
        detailed_df = self.get_detailed_scores()
        
        if df.empty:
            print("No valid results to plot.")
            return
        
        datasets = sorted(df['dataset'].unique())
        
        # Determine number of subplots needed
        n_datasets = len(datasets)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'XGBoost vs Second-Best Model Performance',
                    fontsize=16, fontweight='bold')
        
        # Define consistent colorblind-safe colors for fingerprints across all datasets
        all_fingerprints = sorted(df['fingerprint'].unique())
        
        # Colorblind-safe palette (Wong 2011)
        colorblind_safe_colors = [
            '#E69F00',  # Orange
            '#56B4E9',  # Sky Blue
            '#009E73',  # Bluish Green
            '#F0E442',  # Yellow
            '#0072B2',  # Blue
            '#D55E00',  # Vermillion
            '#CC79A7',  # Reddish Purple
            '#999999'   # Gray
        ]
        
        # Extend with additional colors if needed
        if len(all_fingerprints) > len(colorblind_safe_colors):
            # Fall back to a colorblind-friendly colormap for extra colors
            extra_colors = plt.cm.viridis(np.linspace(0, 1, len(all_fingerprints) - len(colorblind_safe_colors)))
            colorblind_safe_colors.extend(extra_colors)
        
        fingerprint_color_map = {fp: colorblind_safe_colors[i] for i, fp in enumerate(all_fingerprints)}
        
        for idx, dataset in enumerate(datasets):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            dataset_df = df[df['dataset'] == dataset]
            metric_name = dataset_df['metric'].iloc[0]
            is_classification = dataset_df['is_classification'].iloc[0]
            
            # Get models and fingerprints for this dataset
            models = sorted(dataset_df['model'].unique())
            fingerprints = sorted(dataset_df['fingerprint'].unique())
            
            # Create positions for grouped bars
            n_fingerprints = len(fingerprints)
            n_models = len(models)
            
            # Width calculations
            group_width = 0.8
            bar_width = group_width / n_fingerprints
            
            # Position models on x-axis
            model_positions = np.arange(n_models)
            
            # Find best and second-best overall performance
            if is_classification:
                sorted_df = dataset_df.sort_values('mean', ascending=False)
            else:
                sorted_df = dataset_df.sort_values('mean', ascending=True)
            
            best_overall = sorted_df.iloc[0] if len(sorted_df) > 0 else None
            second_best = sorted_df.iloc[1] if len(sorted_df) > 1 else None
            
            # Plot each combination as individual bars with unique colors
            # Create combination color map for this dataset
            dataset_combinations = []
            for _, row in dataset_df.iterrows():
                combo = f"{row['fingerprint']}_{row['model']}"
                if combo not in dataset_combinations:
                    dataset_combinations.append(combo)
            
            combination_color_map = {}
            for i, combo in enumerate(dataset_combinations):
                if i < len(colorblind_safe_colors):
                    combination_color_map[combo] = colorblind_safe_colors[i]
                else:
                    extra_colors_combo = plt.cm.tab20(np.linspace(0, 1, len(dataset_combinations)))
                    combination_color_map[combo] = extra_colors_combo[i]
            
            bar_positions = []
            bar_means = []
            bar_stds = []
            bar_colors = []
            bar_labels = []
            
            position = 0
            for model_idx, model in enumerate(models):
                for fp_idx, fingerprint in enumerate(fingerprints):
                    subset = dataset_df[(dataset_df['fingerprint'] == fingerprint) & 
                                    (dataset_df['model'] == model)]
                    if len(subset) > 0:
                        mean_score = subset['mean'].iloc[0]
                        std_score = subset['std'].iloc[0]
                        combo = f"{fingerprint}_{model}"
                        
                        bar_positions.append(position)
                        bar_means.append(mean_score)
                        bar_stds.append(std_score if pd.notna(std_score) else 0)
                        bar_colors.append(combination_color_map.get(combo, '#999999'))
                        bar_labels.append(model)  # Only show model name
                        position += 1
            
            # Create all bars at once with individual colors
            bars = ax.bar(bar_positions, bar_means, 0.8,
                color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.5,
                yerr=bar_stds, capsize=3, error_kw={'alpha': 0.6})
            
            # Add value labels on top of bars
            for pos, mean_val, std_val in zip(bar_positions, bar_means, bar_stds):
                if mean_val > 0:  # Only if there's data
                    # Position value label above error bar
                    label_height = mean_val + std_val + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                    ax.text(pos, label_height, f'{mean_val:.3f}', ha='center', va='bottom',
                           fontsize=7, color='black', fontweight='normal')
                
            # Add significance stars ONLY on XGBoost bars when it's significantly better
            for bar_idx, (bar, pos, mean_val, std_val, label) in enumerate(zip(bars, bar_positions, bar_means, bar_stds, bar_labels)):
                if 'XGBOOST' in label.upper() and mean_val > 0:
                    # Extract fingerprint and model from label
                    parts = label.split('+')
                    if len(parts) == 2:
                        fingerprint, model = parts[0], parts[1]
                        
                        # Check if this XGBoost is the best performer and compare vs second best
                        if (best_overall is not None and second_best is not None and
                            best_overall['fingerprint'] == fingerprint and 
                            best_overall['model'].upper() == 'XGBOOST'):
                            
                            # Get scores for statistical testing
                            best_data = detailed_df[
                                (detailed_df['dataset'] == dataset) & 
                                (detailed_df['fingerprint'] == best_overall['fingerprint']) &
                                (detailed_df['model'] == best_overall['model'])
                            ]
                            
                            second_data = detailed_df[
                                (detailed_df['dataset'] == dataset) & 
                                (detailed_df['fingerprint'] == second_best['fingerprint']) &
                                (detailed_df['model'] == second_best['model'])
                            ]
                            
                            if len(best_data) > 0 and len(second_data) > 0:
                                best_scores = best_data['score'].values
                                second_scores = second_data['score'].values
                                
                                # Ensure same number of seeds for paired test
                                min_len = min(len(best_scores), len(second_scores))
                                if min_len > 1:
                                    try:
                                        from scipy.stats import ttest_rel
                                        _, p_value = ttest_rel(best_scores[:min_len], second_scores[:min_len])
                                        stars = self.get_significance_stars(p_value)
                                        
                                        if stars:
                                            # Position stars above value labels
                                            star_height = mean_val + std_val + 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                                            ax.text(pos, star_height, stars, ha='center', va='bottom',
                                                   fontsize=16, fontweight='bold', color='red')
                                            
                                            # Add p-value just below stars
                                            p_text = f"p={p_value:.3f}"
                                            ax.text(pos, star_height - 0.002 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                                                   p_text, ha='center', va='top',
                                                   fontsize=8, color='red')
                                    except Exception as e:
                                        print(f"Error in t-test for {dataset}: {e}")
                    current_model = models[bar_idx]
                    current_method = f"{fingerprint}_{current_model}"
                    
                    # Only process XGBoost bars
                    if current_model.upper() == 'XGBOOST' and mean_val > 0:
                        
                        # Check if this XGBoost is the best performer and compare vs second best
                        if (best_overall is not None and second_best is not None and
                            best_overall['fingerprint'] == fingerprint and 
                            best_overall['model'].upper() == 'XGBOOST'):
                            
                            # Get scores for statistical testing
                            best_data = detailed_df[
                                (detailed_df['dataset'] == dataset) & 
                                (detailed_df['fingerprint'] == best_overall['fingerprint']) &
                                (detailed_df['model'] == best_overall['model'])
                            ]
                            
                            second_data = detailed_df[
                                (detailed_df['dataset'] == dataset) & 
                                (detailed_df['fingerprint'] == second_best['fingerprint']) &
                                (detailed_df['model'] == second_best['model'])
                            ]
                            
                            if len(best_data) > 1 and len(second_data) > 1:
                                best_scores = best_data['score'].values
                                second_scores = second_data['score'].values
                                
                                # Ensure same number of seeds for paired test
                                min_len = min(len(best_scores), len(second_scores))
                                if min_len > 1:
                                    try:
                                        from scipy.stats import ttest_rel
                                        _, p_value = ttest_rel(best_scores[:min_len], second_scores[:min_len])
                                        stars = self.get_significance_stars(p_value)
                                        
                                        if stars:
                                            # Position stars above value labels (higher than before)
                                            star_height = mean_val + std_val + 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                                            ax.text(pos, star_height, stars, ha='center', va='bottom',
                                                   fontsize=16, fontweight='bold', color='red')
                                            
                                            # Add p-value below stars
                                            p_text = f"p={p_value:.3f}"
                                            ax.text(pos, star_height - 0.001 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                                                   p_text, ha='center', va='top',
                                                   fontsize=8, color='red')
                                    except Exception as e:
                                        print(f"Error in t-test for {dataset}: {e}")
            
            # Customize the plot
            ax.set_xlabel('Model')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{dataset}')
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(bar_labels, rotation=45, ha='right')
            
            # Add significance legend (no fingerprint legend needed with unique colors)
            if idx == len(datasets) - 1 or len(datasets) == 1:
                textstr = 'XGBoost vs 2nd best\n* p<0.05  ** p<0.01  *** p<0.001'
                props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', bbox=props)
            
            # Set y-axis limits appropriately
            if is_classification:
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(bottom=0)
        
        # Hide unused subplots
        for idx in range(len(datasets), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'xgboost_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary after plotting
        self.print_xgboost_comparison_summary()
    
    def compute_xgboost_significance(self, dataset_name: str, fingerprint: str, detailed_df: pd.DataFrame):
        """Compute statistical significance of XGBoost vs other models for a specific fingerprint"""
        from scipy.stats import ttest_rel
        
        dataset_data = detailed_df[detailed_df['dataset'] == dataset_name]
        fingerprint_data = dataset_data[dataset_data['fingerprint'] == fingerprint]
        
        # Get XGBoost scores
        xgboost_data = fingerprint_data[fingerprint_data['model'] == 'XGBoost']
        if len(xgboost_data) == 0:
            return {}
        
        xgboost_scores = xgboost_data['score'].values
        
        # Compare against other models
        significance_results = {}
        models = fingerprint_data['model'].unique()
        
        for model in models:
            if model == 'XGBoost':
                continue
                
            model_data = fingerprint_data[fingerprint_data['model'] == model]
            if len(model_data) == 0:
                continue
                
            model_scores = model_data['score'].values
            
            # Ensure same number of samples (seeds)
            if len(xgboost_scores) == len(model_scores) and len(xgboost_scores) > 1:
                try:
                    # Use paired t-test
                    _, p_value = ttest_rel(xgboost_scores, model_scores)
                    significance_results[model] = p_value
                except:
                    significance_results[model] = 1.0
            else:
                significance_results[model] = 1.0
        
        return significance_results
        """Create detailed comparison plots grouped by model with fingerprint performance and significance stars"""
        if not self.results:
            print("No results to plot yet. Please run analyze_all_datasets() first.")
            return
        
        # Get summary statistics
        df = self.get_summary_statistics()
        detailed_df = self.get_detailed_scores()
        
        if df.empty:
            print("No valid results to plot.")
            return
        
        # Group datasets by type and sort alphabetically within each group
        classification_datasets = []
        regression_datasets = []
        
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            is_classification = dataset_df['is_classification'].iloc[0]
            
            if is_classification:
                classification_datasets.append(dataset)
            else:
                regression_datasets.append(dataset)
        
        # Sort alphabetically within each group
        classification_datasets = sorted(classification_datasets)
        regression_datasets = sorted(regression_datasets)
        
        # Combine: classification first, then regression
        datasets = classification_datasets + regression_datasets
        
        # Determine number of subplots needed
        n_datasets = len(datasets)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Performance by Dataset and Model (Mean ± Std across seeds)',
                    fontsize=16, fontweight='bold')
        
        # Define consistent colorblind-safe colors for fingerprints across all datasets
        all_fingerprints = sorted(df['fingerprint'].unique())
        
        # Colorblind-safe palette (Wong 2011)
        colorblind_safe_colors = [
            '#E69F00',  # Orange
            '#56B4E9',  # Sky Blue
            '#009E73',  # Bluish Green
            '#F0E442',  # Yellow
            '#0072B2',  # Blue
            '#D55E00',  # Vermillion
            '#CC79A7',  # Reddish Purple
            '#999999'   # Gray
        ]
        
        # Extend with additional colors if needed
        if len(all_fingerprints) > len(colorblind_safe_colors):
            # Fall back to a colorblind-friendly colormap for extra colors
            extra_colors = plt.cm.viridis(np.linspace(0, 1, len(all_fingerprints) - len(colorblind_safe_colors)))
            colorblind_safe_colors.extend(extra_colors)
        
        fingerprint_color_map = {fp: colorblind_safe_colors[i] for i, fp in enumerate(all_fingerprints)}
        
        for idx, dataset in enumerate(datasets):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            dataset_df = df[df['dataset'] == dataset]
            metric_name = dataset_df['metric'].iloc[0]
            is_classification = dataset_df['is_classification'].iloc[0]
            
            # Get models and fingerprints for this dataset
            models = sorted(dataset_df['model'].unique())
            fingerprints = sorted(dataset_df['fingerprint'].unique())
            
            # Create positions for grouped bars
            n_fingerprints = len(fingerprints)
            n_models = len(models)
            
            # Width calculations
            group_width = 0.8
            bar_width = group_width / n_fingerprints
            
            # Calculate model averages
            model_averages = {}
            model_stds = {}
            for model in models:
                model_data = dataset_df[dataset_df['model'] == model]
                model_averages[model] = model_data['mean'].mean()
                model_stds[model] = model_data['mean'].std()
            
            # Position models on x-axis
            model_positions = np.arange(n_models)
            
            # Compute significance for this dataset
            significance_matrix = self.compute_pairwise_significance(dataset, detailed_df)
            
            # Find best method for this dataset
            is_classification = dataset_df['is_classification'].iloc[0]
            if is_classification:
                best_idx = dataset_df['mean'].idxmax()
            else:
                best_idx = dataset_df['mean'].idxmin()
            best_method = f"{dataset_df.loc[best_idx, 'fingerprint']}_{dataset_df.loc[best_idx, 'model']}"
            
            # Create bars for each fingerprint within each model group
            for fp_idx, fingerprint in enumerate(fingerprints):
                fp_means = []
                fp_stds = []
                fp_positions = []
                
                for model_idx, model in enumerate(models):
                    subset = dataset_df[(dataset_df['fingerprint'] == fingerprint) & 
                                    (dataset_df['model'] == model)]
                    if len(subset) > 0:
                        mean_score = subset['mean'].iloc[0]
                        std_score = subset['std'].iloc[0]
                        fp_means.append(mean_score)
                        fp_stds.append(std_score if pd.notna(std_score) else 0)
                    else:
                        fp_means.append(0)
                        fp_stds.append(0)
                    
                    # Calculate position within model group
                    pos = model_positions[model_idx] + (fp_idx - (n_fingerprints-1)/2) * bar_width
                    fp_positions.append(pos)
                
                # Plot bars for this fingerprint across all models with error bars
                bars = ax.bar(fp_positions, fp_means, bar_width * 0.9,
                    label=fingerprint, color=fingerprint_color_map[fingerprint],
                    alpha=0.8, edgecolor='white', linewidth=0.5,
                    yerr=fp_stds, capsize=3, error_kw={'alpha': 0.6})
                
                # Add significance stars on top of bars
                for bar_idx, (bar, pos, mean_val, std_val) in enumerate(zip(bars, fp_positions, fp_means, fp_stds)):
                    if mean_val > 0:  # Only if there's data
                        method = f"{fingerprint}_{models[bar_idx]}"
                        
                        # Check significance against best method
                        if method != best_method and (best_method, method) in significance_matrix:
                            p_value = significance_matrix[(best_method, method)]
                            stars = self.get_significance_stars(p_value)
                            
                            if stars:
                                # Position stars above error bars
                                star_height = mean_val + std_val + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                                ax.text(pos, star_height, stars, ha='center', va='bottom',
                                       fontsize=12, fontweight='bold', color='red')
                        elif method == best_method:
                            # Mark the best method with a crown or special symbol
                            star_height = mean_val + std_val + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                            ax.text(pos, star_height, '👑', ha='center', va='bottom', fontsize=10)
            
            # Add model average lines/markers
            for model_idx, model in enumerate(models):
                avg_score = model_averages[model]
                # Draw a horizontal line across the model group showing average
                left_edge = model_positions[model_idx] - group_width/2
                right_edge = model_positions[model_idx] + group_width/2
                ax.hlines(avg_score, left_edge, right_edge,
                        colors='red', linestyles='--', linewidth=2, alpha=0.7)
                
                # Add average value as text
                y_offset = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.text(model_positions[model_idx], avg_score + y_offset,
                    f'{avg_score:.3f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=8, color='red')
            
            # Customize the plot
            ax.set_xlabel('Model')
            ax.set_ylabel(metric_name)
            
            ax.set_title(f'{dataset}')
            
            ax.set_xticks(model_positions)
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Add vertical lines to separate model groups
            for i in range(1, len(models)):
                ax.axvline(x=model_positions[i] - 0.5, color='gray',
                        linestyle=':', alpha=0.5, linewidth=1)
            
            # Only show legend on first subplot to avoid redundancy
            if idx == 0:
                # Create custom legend with fingerprints and model average
                legend_elements = [plt.Rectangle((0,0),1,1, facecolor=fingerprint_color_map[fp],
                                            alpha=0.8, label=fp) for fp in fingerprints]
                legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--',
                                                linewidth=2, label='Model Average'))
                ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
            
            # Add significance legend on the last subplot or first if only one
            if idx == len(datasets) - 1 or len(datasets) == 1:
                # Add text box with significance explanation
                textstr = 'Significance vs. best:\n* p<0.05  ** p<0.01  *** p<0.001\n👑 = Best method'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', bbox=props)
            
            ax.grid(axis='y', alpha=0.3)
            
            # Set y-axis limits appropriately
            if is_classification:
                ax.set_ylim(0, 1)  # AUROC is bounded between 0 and 1
            else:
                ax.set_ylim(bottom=0)  # RMSE starts from 0
        
        # Hide unused subplots
        for idx in range(len(datasets), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'detailed_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_detailed_significance_table(self):
        """Print detailed significance table for each dataset"""
        if not self.results:
            print("No results available. Please run analyze_all_datasets() first.")
            return
        
        print("\n" + "="*80)
        print("DETAILED SIGNIFICANCE ANALYSIS")
        print("="*80)
        
        df = self.get_summary_statistics()
        detailed_df = self.get_detailed_scores()
        
        for dataset in sorted(self.results.keys()):
            dataset_results = self.results[dataset]
            if not dataset_results or not dataset_results['scores']:
                continue
                
            dataset_df = df[df['dataset'] == dataset]
            metric_name = dataset_results['metric']
            is_classification = dataset_results['is_classification']
            
            print(f"\nDataset: {dataset} ({metric_name})")
            print("-" * 60)
            
            # Sort by performance
            if is_classification:
                dataset_df_sorted = dataset_df.sort_values('mean', ascending=False)
            else:
                dataset_df_sorted = dataset_df.sort_values('mean', ascending=True)
            
            # Get significance matrix
            significance_matrix = self.compute_pairwise_significance(dataset, detailed_df)
            
            # Print ranking table with significance
            print(f"{'Rank':<4} {'Method':<25} {'Score':<12} {'Std':<8} {'Sig vs Best':<12}")
            print("-" * 60)
            
            best_method = None
            for rank, (idx, row) in enumerate(dataset_df_sorted.iterrows(), 1):
                method_name = f"{row['fingerprint']}_{row['model']}"
                if rank == 1:
                    best_method = method_name
                    sig_text = "BEST"
                else:
                    if best_method and (best_method, method_name) in significance_matrix:
                        p_val = significance_matrix[(best_method, method_name)]
                        stars = self.get_significance_stars(p_val)
                        sig_text = f"{stars if stars else 'n.s.'} (p={p_val:.3f})"
                    else:
                        sig_text = "n/a"
                
                print(f"{rank:<4} {row['fingerprint']}+{row['model']:<24} {row['mean']:<8.4f} ±{row['std']:<7.3f} {sig_text:<12}")

    def print_summary(self):
        """Print a summary of the analysis results"""
        if not self.results:
            print("No results available. Please run analyze_all_datasets() first.")
            return
        
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        df = self.get_summary_statistics()
        
        for dataset in sorted(self.results.keys()):
            dataset_results = self.results[dataset]
            if not dataset_results or not dataset_results['scores']:
                continue
                
            dataset_df = df[df['dataset'] == dataset]
            metric_name = dataset_results['metric']
            
            print(f"\nDataset: {dataset} ({metric_name})")
            print("-" * 40)
            
            # Best performing combinations
            if metric_name == "AUROC":
                best_row = dataset_df.loc[dataset_df['mean'].idxmax()]
                print(f"Best: {best_row['fingerprint']} + {best_row['model']} = {best_row['mean']:.4f} ± {best_row['std']:.4f}")
            else:  # RMSE - lower is better
                best_row = dataset_df.loc[dataset_df['mean'].idxmin()]
                print(f"Best: {best_row['fingerprint']} + {best_row['model']} = {best_row['mean']:.4f} ± {best_row['std']:.4f}")
            
            print(f"Number of seeds: {best_row['count']}")
            print(f"Total combinations tested: {len(dataset_df)}")

    def plot_confusion_matrices(self, figsize=(12, 8)):
        """Create professional confusion matrices for XGBoost models using sklearn's ConfusionMatrixDisplay"""
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
        
        # Get classification datasets by checking unique values in targets
        with self.db_manager._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get unique datasets and check if they're classification
            cursor.execute('''
                SELECT DISTINCT dataset_name FROM dataset_targets
            ''')
            datasets = [row[0] for row in cursor.fetchall()]
            
            classification_datasets = []
            for dataset in datasets:
                cursor.execute('''
                    SELECT DISTINCT target_value FROM dataset_targets 
                    WHERE dataset_name = ?
                ''', (dataset,))
                unique_values = [row[0] for row in cursor.fetchall()]
                
                # Consider it classification if all values are 0 or 1
                if all(val in [0.0, 1.0] for val in unique_values) and len(unique_values) > 1:
                    classification_datasets.append(dataset)
        
        if not classification_datasets:
            print("No classification datasets found.")
            return
        
        # Get summary statistics to find best XGBoost models
        summary_df = self.get_summary_statistics()
        xgboost_df = summary_df[summary_df['model'].str.upper() == 'XGBOOST']
        
        if xgboost_df.empty:
            print("No XGBoost results found.")
            return
        
        # Determine subplot layout
        n_datasets = len(classification_datasets)
        n_cols = min(2, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_datasets == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle('Confusion Matrices: Best XGBoost Models', fontsize=16, fontweight='bold', y=0.98)
        
        for idx, dataset in enumerate(classification_datasets):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Find best XGBoost model for this dataset
            dataset_xgb = xgboost_df[xgboost_df['dataset'] == dataset]
            if dataset_xgb.empty:
                ax.text(0.5, 0.5, f'No XGBoost results\nfor {dataset}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(dataset)
                continue
            
            best_xgb = dataset_xgb.loc[dataset_xgb['mean'].idxmax()]
            best_fingerprint = best_xgb['fingerprint']
            best_model = best_xgb['model']
            best_auc = best_xgb['mean']
            
            # Get actual predictions from database
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get all predictions for this model across all seeds to get comprehensive view
                cursor.execute('''
                    SELECT p.prediction, dt.target_value
                    FROM predictions p
                    JOIN dataset_targets dt ON p.dataset_name = dt.dataset_name AND p.data_index = dt.data_index
                    WHERE p.dataset_name = ? AND p.fingerprint = ? AND p.model_name = ?
                ''', (dataset, best_fingerprint, best_model))
                
                data = cursor.fetchall()
            
            if not data:
                ax.text(0.5, 0.5, f'No prediction data\nfor {dataset}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(dataset)
                continue
            
            # Convert to arrays
            y_pred_proba = np.array([row[0] for row in data])
            y_true = np.array([row[1] for row in data]).astype(int)
            
            # Convert probabilities to binary predictions (threshold = 0.5)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            # Create confusion matrix display
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                        display_labels=['Negative', 'Positive'])
            
            # Create custom colormap matching the bar plot colors (Wong 2011 palette)
            from matplotlib.colors import LinearSegmentedColormap
            # Use the sky blue color from our palette for consistency
            colors = ['#ffffff', '#56B4E9']  # White to Sky Blue
            custom_cmap = LinearSegmentedColormap.from_list('custom_blues', colors, N=256)
            
            # Plot with sklearn's display method using matching colors
            disp.plot(ax=ax, cmap=custom_cmap, values_format='d')
            
            # Customize the plot
            ax.set_title(f'{dataset}\n{best_fingerprint} + {best_model}\nAUROC: {best_auc:.3f} | Acc: {accuracy:.3f} | MCC: {mcc:.3f}', 
                        fontsize=11, fontweight='bold')
            
            # Add performance metrics as text
            metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}\nMCC: {mcc:.3f}'
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(len(classification_datasets), len(axes)):
            if idx < len(axes):
                axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'xgboost_confusion_matrices.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"XGBoost confusion matrices saved to: {plot_path}")
        
        # Print summary statistics
        print("\n=== XGBoost Classification Performance Summary ===")
        for dataset in classification_datasets:
            dataset_xgb = xgboost_df[xgboost_df['dataset'] == dataset]
            if not dataset_xgb.empty:
                best_xgb = dataset_xgb.loc[dataset_xgb['mean'].idxmax()]
                print(f"{dataset}: {best_xgb['fingerprint']} + {best_xgb['model']} | AUROC: {best_xgb['mean']:.4f} ± {best_xgb['std']:.4f}")
        print("=" * 50)
        print("MCC (Matthews Correlation Coefficient) ranges from -1 to +1:")
        print("  +1 = perfect prediction")
        print("   0 = no better than random")
        print("  -1 = total disagreement between prediction and observation")

    def plot_roc_curves(self, figsize=(12, 8)):
        """Create ROC curves for XGBoost models using matching color theme"""
        from sklearn.metrics import roc_curve, auc
        
        # Get classification datasets by checking unique values in targets
        with self.db_manager._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get unique datasets and check if they're classification
            cursor.execute('''
                SELECT DISTINCT dataset_name FROM dataset_targets
            ''')
            datasets = [row[0] for row in cursor.fetchall()]
            
            classification_datasets = []
            for dataset in datasets:
                cursor.execute('''
                    SELECT DISTINCT target_value FROM dataset_targets 
                    WHERE dataset_name = ?
                ''', (dataset,))
                unique_values = [row[0] for row in cursor.fetchall()]
                
                # Consider it classification if all values are 0 or 1
                if all(val in [0.0, 1.0] for val in unique_values) and len(unique_values) > 1:
                    classification_datasets.append(dataset)
        
        if not classification_datasets:
            print("No classification datasets found.")
            return
        
        # Get summary statistics to find best XGBoost models
        summary_df = self.get_summary_statistics()
        xgboost_df = summary_df[summary_df['model'].str.upper() == 'XGBOOST']
        
        if xgboost_df.empty:
            print("No XGBoost results found.")
            return
        
        # Determine subplot layout
        n_datasets = len(classification_datasets)
        n_cols = min(2, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_datasets == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle('ROC Curves: Best XGBoost Models', fontsize=16, fontweight='bold', y=0.98)
        
        # Use the same colorblind-safe color from our palette (sky blue)
        main_color = '#56B4E9'  # Sky Blue from Wong 2011 palette
        
        for idx, dataset in enumerate(classification_datasets):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Find best XGBoost model for this dataset
            dataset_xgb = xgboost_df[xgboost_df['dataset'] == dataset]
            if dataset_xgb.empty:
                ax.text(0.5, 0.5, f'No XGBoost results\nfor {dataset}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(dataset)
                continue
            
            best_xgb = dataset_xgb.loc[dataset_xgb['mean'].idxmax()]
            best_fingerprint = best_xgb['fingerprint']
            best_model = best_xgb['model']
            best_auc = best_xgb['mean']
            
            # Get actual predictions from database
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get all predictions for this model across all seeds
                cursor.execute('''
                    SELECT p.prediction, dt.target_value
                    FROM predictions p
                    JOIN dataset_targets dt ON p.dataset_name = dt.dataset_name AND p.data_index = dt.data_index
                    WHERE p.dataset_name = ? AND p.fingerprint = ? AND p.model_name = ?
                ''', (dataset, best_fingerprint, best_model))
                
                data = cursor.fetchall()
            
            if not data:
                ax.text(0.5, 0.5, f'No prediction data\nfor {dataset}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(dataset)
                continue
            
            # Convert to arrays
            y_pred_proba = np.array([row[0] for row in data])
            y_true = np.array([row[1] for row in data]).astype(int)
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color=main_color, linewidth=2.5, 
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
            
            # Plot diagonal line (random classifier)
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.8, 
                   linewidth=1.5, label='Random classifier')
            
            # Customize plot
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontsize=11)
            ax.set_title(f'{dataset}\n{best_fingerprint} + {best_model}', 
                        fontsize=11, fontweight='bold')
            ax.legend(loc="lower right", fontsize=9)
            
            # Add some styling to match our theme
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('gray')
            ax.spines['bottom'].set_color('gray')
        
        # Hide unused subplots
        for idx in range(len(classification_datasets), len(axes)):
            if idx < len(axes):
                axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'xgboost_roc_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"XGBoost ROC curves saved to: {plot_path}")
        
        # Print ROC summary
        print("\n=== XGBoost ROC Curve Summary ===")
        for dataset in classification_datasets:
            dataset_xgb = xgboost_df[xgboost_df['dataset'] == dataset]
            if not dataset_xgb.empty:
                best_xgb = dataset_xgb.loc[dataset_xgb['mean'].idxmax()]
                print(f"{dataset}: {best_xgb['fingerprint']} + {best_xgb['model']} | AUROC: {best_xgb['mean']:.4f} ± {best_xgb['std']:.4f}")
        print("=" * 40)

# Example usage:
def run_analysis(db_manager, save_dir="./analysis_results"):
    """Run complete analysis pipeline"""
    analyzer = BenchmarkManager(db_manager, save_dir)
    
    # Analyze all datasets
    analyzer.analyze_all_datasets()
    
    # Run statistical analysis
    analyzer.run_statistical_analysis()
    
    # Print winners summary
    analyzer.print_winners_summary()
    
    # Create XGBoost vs second-best comparison plot with significance stars
    analyzer.plot_xgboost_vs_second_best()
    
    # Create confusion matrices for classification datasets
    analyzer.plot_confusion_matrices()
    
    # Create ROC curves for classification datasets
    analyzer.plot_roc_curves()
    
    # Print detailed significance analysis
    analyzer.print_detailed_significance_table()
    
    # Save results
    #analyzer.save_results()
    
    # Print summary
    analyzer.print_summary()
    
    return analyzer

if __name__ == '__main__':
    master_job_id = sys.argv[1]
    base = Path('./studies')
    path = base / master_job_id
    directories = [item.name for item in path.iterdir() if item.is_dir()]
    for dic in directories:
        subdir_path = path / dic
        string = f"{subdir_path}/predictions.db"
        print(f"Processing {string}")
        db_manager = DatabaseManager(string)
        analyzer = run_analysis(db_manager, str(subdir_path))