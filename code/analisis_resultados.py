import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ResultadosAnalyzer:
    def __init__(self, results_path="results"):
        self.results_path = Path(results_path)
        self.data = {}
        self.df_combined = None
    
    def load_all_experiments(self):
        for exp_path in self.results_path.iterdir():
            if exp_path.is_dir():
                exp_name = exp_path.name
                self.load_experiment(exp_name, exp_path)
    
    def load_experiment(self, exp_name, exp_path):
        try:
            if exp_name == "double_descent_analysis":
                self.load_double_descent_data(exp_path)
            else:
                main_file = exp_path / "experiment_results_full.json"
                if main_file.exists():
                    with open(main_file, 'r') as f:
                        self.data[exp_name] = json.load(f)
            else:
                self.data[exp_name] = {}
                    for model_dir in exp_path.iterdir():
                        if model_dir.is_dir():
                            model_file = model_dir / "evaluation_metrics_full.json"
                            if model_file.exists():
                                with open(model_file, 'r') as f:
                                    model_data = json.load(f)
                                    self.data[exp_name][model_dir.name] = model_data
        except Exception as e:
            print(f"Error loading {exp_name}: {e}")

    def load_double_descent_data(self, exp_path):
        self.data["double_descent_analysis"] = {}
        for file_path in exp_path.glob("*.json"):
            with open(file_path, 'r') as f:
                self.data["double_descent_analysis"][file_path.stem] = json.load(f)
    
    def extract_all_data(self):
        records = []
        
        for exp_name, exp_data in self.data.items():
            if exp_name == "double_descent_analysis":
                records.extend(self.extract_double_descent_data(exp_data, exp_name))
            else:
                records.extend(self.extract_generic_data(exp_data, exp_name))
        
        self.df_combined = pd.DataFrame(records)
        self.clean_and_normalize_data()
    
    def extract_generic_data(self, exp_data, exp_name):
        records = []
        
        if isinstance(exp_data, dict) and "model_results" in exp_data:
            model_results = exp_data["model_results"]
            for model_name, model_data in model_results.items():
                if isinstance(model_data, dict) and "user_group_results" in model_data:
                    for user_group, group_data in model_data["user_group_results"].items():
                        record = self.create_base_record(exp_name, model_name, user_group, group_data)
                        records.append(record)
        
        return records
    
    def extract_double_descent_data(self, exp_data, exp_name):
        records = []
        for model_name, model_data in exp_data.items():
            if "user_group_results" in model_data:
                for user_group, group_data in model_data["user_group_results"].items():
                    record = self.create_base_record(exp_name, model_name, user_group, group_data)
                    records.append(record)
        return records
    
    def create_base_record(self, exp_name, model_name, user_group, group_data):
        approach = self.get_approach_from_model(model_name)
        
        feature_importance = group_data.get("feature_importance", {})
        
        record = {
            "experiment": exp_name,
            "model_name": model_name,
            "user_group": user_group,
            "approach": approach,
            "complexity_level": group_data.get("complexity_level", "unknown"),
            "ndcg_at_k": group_data.get("ndcg_at_k", 0),
            "precision_at_k": group_data.get("precision_at_k", 0),
            "recall_at_k": group_data.get("recall_at_k", 0),
            "coverage": group_data.get("coverage", 0),
            "num_users_evaluated": group_data.get("num_users_evaluated", 0),
            "feature_importance": feature_importance,
            "feature_names": group_data.get("model_info", {}).get("feature_names", []),
            "feature_count": group_data.get("model_info", {}).get("feature_count", 0)
        }
        
        return record
    
    def get_approach_from_model(self, model_name):
        if "content" in model_name:
            return "content"
        elif "collaborative" in model_name:
            return "collaborative"
        elif "neural" in model_name:
            return "neural"
        elif "hybrid" in model_name:
            return "hybrid"
        else:
        return "unknown"

    def clean_and_normalize_data(self):
        if self.df_combined is None or self.df_combined.empty:
            return
        
        self.df_combined = self.df_combined.dropna(subset=['ndcg_at_k', 'precision_at_k', 'recall_at_k'])
        self.df_combined = self.df_combined[self.df_combined['approach'] != 'unknown']
    
    def calculate_explainability_metrics(self):
        if self.df_combined is None or self.df_combined.empty:
            return
        
        self.df_combined['simplicity_score'] = self.df_combined.apply(self.calculate_simplicity, axis=1)
        self.df_combined['interpretability_score'] = self.df_combined.apply(self.calculate_interpretability, axis=1)
        self.df_combined['concentration_score'] = self.df_combined.apply(self.calculate_concentration, axis=1)
        self.df_combined['explainability_score'] = (
            self.df_combined['simplicity_score'] * 0.4 +
            self.df_combined['interpretability_score'] * 0.4 +
            self.df_combined['concentration_score'] * 0.2
        )
    
    def calculate_simplicity(self, row):
        feature_importance = row['feature_importance']
        if not feature_importance:
            return 0.0
        
        values = list(feature_importance.values())
        if isinstance(values[0], list):
            values = [v for sublist in values for v in sublist]
        
        sorted_values = sorted(values, reverse=True)
        cumulative = np.cumsum(sorted_values)
        threshold = 0.9 * sum(sorted_values)
        
        if threshold == 0:
            return 0.0
        
        num_features = np.argmax(cumulative >= threshold) + 1
        return 1.0 / (1.0 + num_features)
    
    def calculate_interpretability(self, row):
        feature_names = row['feature_names']
        if not feature_names:
            return 0.0
        
        interpretable_features = [f for f in feature_names if any(keyword in f for keyword in ['genre_', 'age_', 'occupation_', 'gender_'])]
        return len(interpretable_features) / len(feature_names)
    
    def calculate_concentration(self, row):
        feature_importance = row['feature_importance']
        if not feature_importance:
            return 0.0
        
        values = list(feature_importance.values())
        if isinstance(values[0], list):
            values = [v for sublist in values for v in sublist]
        
        if sum(values) == 0:
            return 0.0
        
        return max(values) / sum(values)
    
    def plot_performance_vs_explainability(self):
        if self.df_combined is None or self.df_combined.empty:
            print("No data available for plotting")
            return
        
        # Remove corrupted data (neural_transformer with 0.0 performance)
        clean_data = self.df_combined[
            ~((self.df_combined['model_name'] == 'neural_transformer') & 
              (self.df_combined['ndcg_at_k'] == 0.0))
        ].copy()
        
        plt.figure(figsize=(12, 8))
        
        # Professional color palette
        color_palette = {
            'content': '#1f77b4',      # Blue
            'collaborative': '#ff7f0e', # Orange
            'neural': '#2ca02c',       # Green
            'hybrid': '#d62728'        # Red
        }
        
        # Track what we've plotted for legend
        plotted_approaches = set()
        
        for approach in clean_data['approach'].unique():
            subset = clean_data[clean_data['approach'] == approach]
            
            if not subset.empty:
                scatter = plt.scatter(
                    subset['ndcg_at_k'],
                    subset['explainability_score'],
            s=120,
                    c=color_palette[approach],
                    alpha=0.7,
                    label=approach.capitalize(),
                    edgecolors='white',
                    linewidth=1.5,
                    zorder=3
                )
                plotted_approaches.add(approach)
        
        # Add subtle quadrant lines only
        plt.axhline(y=0.35, color='gray', linestyle='--', alpha=0.5, zorder=1)
        plt.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, zorder=1)
        
        plt.xlabel('Performance (NDCG@K)', fontsize=14, fontweight='bold')
        plt.ylabel('Explainability Score', fontsize=14, fontweight='bold')
        plt.title('Trade-off between Performance and Explainability in Recommendation Systems', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(True, alpha=0.3, zorder=1)
        
        # Create horizontal legend below x-axis
        handles = []
        labels = []
        for approach in ['content', 'collaborative', 'neural', 'hybrid']:
            if approach in plotted_approaches:
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color_palette[approach], 
                                        markersize=10, markeredgecolor='black'))
                labels.append(approach.capitalize())
        
        plt.legend(handles, labels, title='Approach', 
                  title_fontsize=12, fontsize=11, 
                  loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                  ncol=4, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        plots_dir = Path("results/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / 'performance_vs_explainability.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Plot 1 saved: performance_vs_explainability.png")
        
        # Print academic summary
        print(f"\n📊 ACADEMIC SUMMARY:")
        print(f"Models analyzed: {len(clean_data)} (excluded {len(self.df_combined) - len(clean_data)} corrupted)")
        print(f"Approaches: {', '.join(plotted_approaches)}")
        print(f"Performance range: {clean_data['ndcg_at_k'].min():.3f} - {clean_data['ndcg_at_k'].max():.3f}")
        print(f"Explainability range: {clean_data['explainability_score'].min():.3f} - {clean_data['explainability_score'].max():.3f}")

    def plot_complexity_vs_performance(self):
        if self.df_combined is None or self.df_combined.empty:
            print("No data available for plotting")
            return
        
        # Remove corrupted data (neural_transformer with 0.0 performance)
        clean_data = self.df_combined[
            ~((self.df_combined['model_name'] == 'neural_transformer') & 
              (self.df_combined['ndcg_at_k'] == 0.0))
        ].copy()
        
        # Create complexity score (low=1, medium=2, high=3)
        complexity_mapping = {'low': 1, 'medium': 2, 'high': 3}
        clean_data['complexity_score'] = clean_data['complexity_level'].map(complexity_mapping)
        
        plt.figure(figsize=(12, 8))
        
        # Professional color palette
        color_palette = {
            'content': '#1f77b4',      # Blue
            'collaborative': '#ff7f0e', # Orange
            'neural': '#2ca02c',       # Green
            'hybrid': '#d62728'        # Red
        }
        
        # Track what we've plotted for legend
        plotted_approaches = set()
        
        for approach in clean_data['approach'].unique():
            subset = clean_data[clean_data['approach'] == approach]
            
            if not subset.empty:
                scatter = plt.scatter(
                    subset['complexity_score'],
                    subset['ndcg_at_k'],
                    s=120,
                    c=color_palette[approach],
                    alpha=0.7,
                    label=approach.capitalize(),
                    edgecolors='white',
                    linewidth=1.5,
                    zorder=3
                )
                plotted_approaches.add(approach)
        
        # Add trend line
        z = np.polyfit(clean_data['complexity_score'], clean_data['ndcg_at_k'], 1)
        p = np.poly1d(z)
        plt.plot(clean_data['complexity_score'], p(clean_data['complexity_score']), 
                "r--", alpha=0.8, linewidth=2, zorder=2)
        
        # Calculate R-squared
        correlation = np.corrcoef(clean_data['complexity_score'], clean_data['ndcg_at_k'])[0, 1]
        r_squared = correlation ** 2
        
        plt.xlabel('Model Complexity', fontsize=14, fontweight='bold')
        plt.ylabel('Performance (NDCG@K)', fontsize=14, fontweight='bold')
        plt.title('Relationship between Model Complexity and Performance in Recommendation Systems', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Set x-axis ticks
        plt.xticks([1, 2, 3], ['Low', 'Medium', 'High'], fontsize=12)
        
        plt.grid(True, alpha=0.3, zorder=1)
        
        # Add R-squared annotation
        plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Create horizontal legend below x-axis
        handles = []
        labels = []
        for approach in ['content', 'collaborative', 'neural', 'hybrid']:
            if approach in plotted_approaches:
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color_palette[approach], 
                                        markersize=10, markeredgecolor='black'))
                labels.append(approach.capitalize())
        
        plt.legend(handles, labels, title='Approach', 
                  title_fontsize=12, fontsize=11, 
                  loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                  ncol=4, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        plots_dir = Path("results/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / 'complexity_vs_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Plot 2 saved: complexity_vs_performance.png")
        
        # Print academic summary
        print(f"\n📊 PLOT 2 SUMMARY:")
        print(f"Models analyzed: {len(clean_data)} (excluded {len(self.df_combined) - len(clean_data)} corrupted)")
        print(f"Approaches: {', '.join(plotted_approaches)}")
        print(f"Complexity levels: {clean_data['complexity_level'].value_counts().to_dict()}")
        print(f"Performance range: {clean_data['ndcg_at_k'].min():.3f} - {clean_data['ndcg_at_k'].max():.3f}")
        print(f"R-squared: {r_squared:.3f}")

    def plot_double_descent_analysis(self):
        if self.df_combined is None or self.df_combined.empty:
            print("No data available for plotting")
            return
        
        # Load specific double descent data
        double_descent_file = Path("results/double_descent_analysis/double_descent_results.json")
        if not double_descent_file.exists():
            print("Double descent results file not found")
            return
        
        with open(double_descent_file, 'r') as f:
            double_descent_data = json.load(f)
        
        plt.figure(figsize=(12, 8))
        
        # Professional color palette
        color_palette = {
            'content_rf': '#1f77b4',      # Blue
            'collaborative_svd': '#ff7f0e', # Orange
            'neural_ncf': '#2ca02c'       # Green
        }
        
        # Track what we've plotted for legend
        plotted_models = set()
        
        for model_name, model_results in double_descent_data.items():
            if not model_results:
                continue
            
            # Extract complexity values and performance
            complexity_values = [result['complexity_value'] for result in model_results]
            performance_values = [result['ndcg_at_k'] for result in model_results]
            
            # Get approach for color
            approach = self.get_approach_from_model(model_name)
            color = color_palette.get(model_name, '#d62728')
            
            # Plot the progression
            scatter = plt.scatter(
                complexity_values,
                performance_values,
                s=120,
                c=color,
                alpha=0.7,
                label=model_name.replace('_', ' ').title(),
                edgecolors='white',
                linewidth=1.5,
                zorder=3
            )
            plotted_models.add(model_name)
            
            # Add trend line for this model
            if len(complexity_values) > 2:
                z = np.polyfit(complexity_values, performance_values, 2)
                p = np.poly1d(z)
                
                # Create smooth curve for this model
                x_smooth = np.linspace(min(complexity_values), max(complexity_values), 100)
                y_smooth = p(x_smooth)
                
                plt.plot(x_smooth, y_smooth, color=color, linestyle='--', 
                        alpha=0.6, linewidth=2, zorder=2)
        
        plt.xlabel('Complexity Parameter Value', fontsize=14, fontweight='bold')
        plt.ylabel('Performance (NDCG@K)', fontsize=14, fontweight='bold')
        plt.title('Double Descent Analysis: Performance vs Complexity Parameter', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(True, alpha=0.3, zorder=1)
        
        # Create horizontal legend below x-axis
        handles = []
        labels = []
        for model_name in ['content_rf', 'collaborative_svd', 'neural_ncf']:
            if model_name in plotted_models:
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color_palette[model_name], 
                                        markersize=10, markeredgecolor='black'))
                labels.append(model_name.replace('_', ' ').title())
        
        plt.legend(handles, labels, title='Model', 
                  title_fontsize=12, fontsize=11, 
                  loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                  ncol=3, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        plots_dir = Path("results/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / 'double_descent_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Plot 3 saved: double_descent_analysis.png")
        
        # Print academic summary
        print(f"\n📊 PLOT 3 SUMMARY:")
        print(f"Models analyzed: {len(plotted_models)}")
        print(f"Models: {', '.join([m.replace('_', ' ').title() for m in plotted_models])}")
        
        # Print double descent patterns
        for model_name, model_results in double_descent_data.items():
            if model_results:
                complexity_values = [result['complexity_value'] for result in model_results]
                performance_values = [result['ndcg_at_k'] for result in model_results]
                param_name = model_results[0]['complexity_param']
                
                print(f"\n{model_name.replace('_', ' ').title()}:")
                print(f"  Parameter: {param_name}")
                print(f"  Complexity range: {min(complexity_values)} - {max(complexity_values)}")
                print(f"  Performance range: {min(performance_values):.3f} - {max(performance_values):.3f}")
                
                # Check for double descent pattern
                if len(performance_values) >= 3:
                    # Simple check: if performance drops then recovers
                    max_perf = max(performance_values)
                    min_perf = min(performance_values)
                    max_idx = performance_values.index(max_perf)
                    min_idx = performance_values.index(min_perf)
                    
                    if max_idx < min_idx:
                        print(f"  Pattern: Double Descent detected (peak at {complexity_values[max_idx]}, valley at {complexity_values[min_idx]})")
                    else:
                        print(f"  Pattern: No clear double descent")

    def plot_user_group_analysis(self):
        """Plot 4: User Group Analysis - Heatmap showing performance by approach vs user group."""
        if self.df_combined is None or self.df_combined.empty:
            print("No data available for plotting")
            return
        
        # Filter data to include only user group experiments (exclude 'full' dataset)
        user_group_data = self.df_combined[
            (self.df_combined['user_group'].isin(['moderate_users', 'active_users', 'power_users'])) &
            (self.df_combined['ndcg_at_k'].notna())
        ].copy()
        
        if user_group_data.empty:
            print("No user group data available for plotting")
            return
        
        # Create pivot table for heatmap: approach vs user_group, values = mean NDCG
        heatmap_data = user_group_data.pivot_table(
            values='ndcg_at_k',
            index='approach',
            columns='user_group',
            aggfunc='mean'
        )
        
        # Reorder columns for logical progression
        column_order = ['moderate_users', 'active_users', 'power_users']
        heatmap_data = heatmap_data.reindex(columns=column_order)
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        
        # Professional color palette for heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (higher = better)
            center=heatmap_data.values.mean(),
            square=True,
            cbar_kws={'label': 'NDCG@K Score', 'shrink': 0.8},
            linewidths=0.5,
            linecolor='white',
            annot_kws={'size': 11, 'weight': 'bold'}
        )
        
        plt.title('Model Robustness Across User Activity Levels\nPerformance (NDCG@K) by Approach and User Group', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('User Activity Level', fontsize=14, fontweight='bold')
        plt.ylabel('Recommendation Approach', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        plots_dir = Path("results/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / 'user_group_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Plot 4 saved: user_group_analysis.png")
        
        # Print academic summary
        print(f"\n📊 PLOT 4 SUMMARY:")
        print(f"User groups analyzed: {len(column_order)}")
        print(f"Approaches analyzed: {len(heatmap_data.index)}")
        print(f"Models per approach: {user_group_data.groupby('approach').size().to_dict()}")
        
        # Calculate robustness metrics
        print(f"\nRobustness Analysis:")
        for approach in heatmap_data.index:
            approach_performance = heatmap_data.loc[approach]
            performance_range = approach_performance.max() - approach_performance.min()
            performance_std = approach_performance.std()
            mean_performance = approach_performance.mean()
            
            print(f"\n{approach.replace('_', ' ').title()}:")
            print(f"  Mean performance: {mean_performance:.3f}")
            print(f"  Performance range: {performance_range:.3f}")
            print(f"  Performance std: {performance_std:.3f}")
            print(f"  Robustness score: {mean_performance/performance_std:.2f} (higher = more robust)")
            
            # Identify best and worst user groups
            best_group = approach_performance.idxmax()
            worst_group = approach_performance.idxmin()
            print(f"  Best for: {best_group.replace('_', ' ').title()} ({approach_performance[best_group]:.3f})")
            print(f"  Worst for: {worst_group.replace('_', ' ').title()} ({approach_performance[worst_group]:.3f})")

    def plot_feature_importance_analysis(self):
        """Plot 5: Feature Importance Analysis - 1x3 layout with proper spacing."""
        if self.df_combined is None or self.df_combined.empty:
            print("No data available for plotting")
            return
        
        # Filter data to include only models with feature importance data
        feature_data = self.df_combined[
            (self.df_combined['feature_importance'].notna()) &
            (self.df_combined['feature_importance'].str.len() > 0)
        ].copy()
        
        if feature_data.empty:
            print("No feature importance data available for plotting")
            return
        
        # Get approaches (excluding neural)
        approaches = [app for app in feature_data['approach'].unique() if app != 'neural']
        
        # Create subplots with more space for title
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        axes = axes.flatten()
        
        # Professional color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, approach in enumerate(approaches):
            ax = axes[idx]
            approach_data = feature_data[feature_data['approach'] == approach]
            
            # Aggregate feature importance across all models in this approach
            all_features = {}
            for _, row in approach_data.iterrows():
                if isinstance(row['feature_importance'], dict):
                    for feature, importance in row['feature_importance'].items():
                        if feature not in all_features:
                            all_features[feature] = []
                        all_features[feature].append(importance)
            
            # Calculate mean importance for each feature
            feature_means = {}
            for feature, values in all_features.items():
                if values:
                    numeric_values = [v for v in values if v is not None and isinstance(v, (int, float))]
                    if numeric_values:
                        feature_means[feature] = np.mean(numeric_values)
            
            # Sort features by importance and get top 10
            sorted_features = sorted(feature_means.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if not sorted_features:
                ax.text(0.5, 0.5, f'No feature data\nfor {approach}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{approach.replace("_", " ").title()} Features', fontsize=14, fontweight='bold')
                continue
            
            features, importances = zip(*sorted_features)
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(features)), importances, color=colors[idx], alpha=0.8, height=0.6)
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, importances)):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{importance:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
            
            # Customize plot
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels([f.replace('_', ' ').title() for f in features], fontsize=10)
            ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
            ax.set_title(f'{approach.replace("_", " ").title()} Features', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Remove margins and set tight x-axis limits
            ax.set_xlim(0, max(importances) * 1.1)  # Add 10% margin to max value
            
            # Invert y-axis for better readability
            ax.invert_yaxis()
        
        # Main title with more space
        plt.suptitle('Feature Importance Analysis by Recommendation Approach\nTop 10 Most Important Features per Approach', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, wspace=0.6, bottom=0.15)
        
        plots_dir = Path("results/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / 'feature_importance_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Plot 5 saved: feature_importance_analysis.png")
        
        # Print academic summary
        print(f"\n📊 PLOT 5 SUMMARY:")
        print(f"Approaches analyzed: {len(approaches)}")
        print(f"Models with feature data: {len(feature_data)}")
        
        # Analyze feature patterns by approach
        print(f"\nFeature Importance Analysis:")
        for approach in approaches:
            approach_data = feature_data[feature_data['approach'] == approach]
            print(f"\n{approach.replace('_', ' ').title()}:")
            print(f"  Models with feature data: {len(approach_data)}")
            
            # Get top features for this approach
            all_features = {}
            for _, row in approach_data.iterrows():
                if isinstance(row['feature_importance'], dict):
                    for feature, importance in row['feature_importance'].items():
                        if feature not in all_features:
                            all_features[feature] = []
                        all_features[feature].append(importance)
            
            if all_features:
                feature_means = {}
                for f, v in all_features.items():
                    numeric_values = [val for val in v if val is not None and isinstance(val, (int, float))]
                    if numeric_values:
                        feature_means[f] = np.mean(numeric_values)
                
                if feature_means:
                    top_features = sorted(feature_means.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    print(f"  Top 5 features:")
                    for feature, importance in top_features:
                        print(f"    {feature.replace('_', ' ').title()}: {importance:.3f}")
                    
                    # Calculate feature diversity
                    importance_values = list(feature_means.values())
                    if importance_values:
                        print(f"  Feature importance range: {min(importance_values):.3f} - {max(importance_values):.3f}")
                        print(f"  Feature importance std: {np.std(importance_values):.3f}")
                else:
                    print(f"  No valid feature importance data")

    def print_summary(self):
        if self.df_combined is None or self.df_combined.empty:
            print("No data available")
            return
        
        print(f"Total models: {len(self.df_combined)}")
        print(f"Experiments: {self.df_combined['experiment'].unique()}")
        print(f"Approaches: {self.df_combined['approach'].unique()}")
        print(f"User groups: {self.df_combined['user_group'].unique()}")
        
        print("\nPerformance by approach:")
        for approach in self.df_combined['approach'].unique():
            subset = self.df_combined[self.df_combined['approach'] == approach]
            print(f"{approach}: NDCG={subset['ndcg_at_k'].mean():.3f}, Explainability={subset['explainability_score'].mean():.3f}")

if __name__ == "__main__":
    analyzer = ResultadosAnalyzer()
    analyzer.load_all_experiments()
    analyzer.extract_all_data()
    analyzer.calculate_explainability_metrics()
    analyzer.print_summary()
    analyzer.plot_performance_vs_explainability()
    analyzer.plot_complexity_vs_performance()
    analyzer.plot_double_descent_analysis()
    analyzer.plot_user_group_analysis()
    analyzer.plot_feature_importance_analysis()