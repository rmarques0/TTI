#!/usr/bin/env python3
"""
TTI Recommendation System - Flexible Experiment Runner v2.0

This script runs experiments across different recommendation approaches:
- Content-based filtering
- Collaborative filtering  
- Hybrid approaches

Features:
- Modular architecture supporting easy addition of new models
- Configurable experiments via config.py
- User group analysis (cold start, warm start, main users)
- Double descent hypothesis testing
- Model explainability analysis
- Comprehensive evaluation metrics

Usage:
    python run_experiments_v2.py                    # Run all experiments + double descent
    python run_experiments_v2.py --skip_double_descent  # Run all experiments except double descent
    python run_experiments_v2.py --experiment cross_approach_comparison
    python run_experiments_v2.py --approach content
    python run_experiments_v2.py --complexity low
    python run_experiments_v2.py --models content_logistic,collaborative_svd
"""

import sys
import os
import argparse
import importlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import json
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from core.data_manager import DataManager
from core.global_data_manager import global_data_manager
from core.base_recommender import BaseRecommender

class ExperimentRunner:
    """
    Main experiment runner for TTI recommendation system.
    Handles model loading, training, evaluation, and result saving.
    """
    
    def __init__(self, args_overrides: Dict = None):
        """
        Initialize experiment runner with configuration.
        
        Args:
            args_overrides: Command-line argument overrides
        """
        # Build config from the config module to ensure all settings are loaded
        self.config = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        
        # Apply command-line argument overrides if any
        if args_overrides:
            for key, value in args_overrides.items():
                if key.upper() in self.config and value is not None:
                    self.config[key.upper()] = value
                    if key == 'sample_size' and value is not None:
                         print(f"Override: Using sample size = {value}")
        
        # Use global data manager for consistency
        self.data_manager = None
        self.results = {}
        self.experiment_start_time = datetime.now()
        
        # Initialize global data manager with current config
        global_data_manager.config.update(self.config)
        global_data_manager.force_rebuild = args_overrides.get('force_rebuild', False) if args_overrides else False
        
    def load_model_class(self, model_name: str) -> BaseRecommender:
        """
        Dynamically load model class from registry.
        
        Args:
            model_name: Name of model to load
            
        Returns:
            Instantiated model object
        """
        if model_name not in config.MODEL_REGISTRY:
            # Check legacy mapping
            if model_name in config.LEGACY_MODEL_MAPPING:
                model_name = config.LEGACY_MODEL_MAPPING[model_name]
            else:
                raise ValueError(f"Unknown model: {model_name}")
        
        model_config = config.MODEL_REGISTRY[model_name]
        class_path = model_config["class_path"]
        
        # Parse module and class name
        module_path, class_name = class_path.rsplit('.', 1)
        
        # Import module and get class
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            
            # Instantiate with parameters
            params = model_config.get("params", {})
            
            # Neural and Hybrid models expect parameters in config dict, others accept direct kwargs
            if module_path.endswith('neural') or module_path.endswith('hybrid'):
                model_instance = model_class(config=params)
            else:
                model_instance = model_class(**params)
            
            return model_instance
            
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not load model {model_name}: {e}")
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load and prepare data for experiments using global data manager.
        
        Returns:
            Tuple of (train_data, test_data, user_groups)
        """
        print("Loading and preprocessing data (using global splits)...")
        
        # Get consistent data from global manager
        train_data, test_data, user_groups = global_data_manager.get_data_for_experiment()
        
        print(f"Data prepared: {len(train_data)} training, {len(test_data)} testing samples")
        print(f"User groups: {[(group, info['count']) for group, info in user_groups.items()]}")
        
        return train_data, test_data, user_groups
    
    def filter_data_by_user_group(self, data: pd.DataFrame, user_group: str) -> pd.DataFrame:
        """
        Filter data to specific user group using global data manager.
        
        Args:
            data: Input dataframe
            user_group: User group name ('cold_start', 'warm_start', 'main_user', 'full')
            
        Returns:
            Filtered dataframe
        """
        return global_data_manager.filter_data_by_user_group(data, user_group)
    
    def evaluate_model(self, model: BaseRecommender, test_data: pd.DataFrame, 
                      k: int = 10) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model instance
            test_data: Test dataset
            k: Number of recommendations
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Generate predictions
        predictions = model.predict(test_data, k=k)
        
        # Create ground truth from test data
        ground_truth = {}
        for user_id in test_data['user_id'].unique():
            user_data = test_data[test_data['user_id'] == user_id]
            liked_items = user_data[user_data['is_positive'] == 1]['movie_id'].tolist()
            ground_truth[user_id] = liked_items
        
        # Calculate metrics
        metrics = model.evaluate(predictions, ground_truth)
        
        # Add model-specific metrics
        metrics.update({
            'model_name': model.name,
            'complexity_level': model.complexity_level,
            'approach_type': model.approach_type,
            'feature_importance': model.get_feature_importance(),
            'model_info': model.get_model_info()
        })
        
        return metrics
    
    def run_single_experiment(self, model_name: str, train_data: pd.DataFrame, 
                            test_data: pd.DataFrame, user_groups: List[str]) -> Dict[str, Any]:
        """
        Run experiment for a single model across multiple user groups.
        
        Args:
            model_name: Name of model to test
            train_data: Training dataset
            test_data: Test dataset  
            user_groups: List of user groups to evaluate
            
        Returns:
            Dictionary of results by user group
        """
        print(f"Running experiment: {model_name}")
        
        # Load and train model
        model = self.load_model_class(model_name)
        
        start_time = time.time()
        model.fit(train_data)
        training_time = time.time() - start_time
        
        # Evaluate on each user group
        results = {
            'model_name': model_name,
            'training_time': training_time,
            'user_group_results': {}
        }
        
        for user_group in user_groups:
            print(f"  Evaluating on {user_group} users...")
            
            # Filter test data for user group
            group_test_data = self.filter_data_by_user_group(test_data, user_group)
            
            if len(group_test_data) == 0:
                print(f"  No data for {user_group} group, skipping...")
                continue
            
            # Evaluate model
            start_eval_time = time.time()
            group_metrics = self.evaluate_model(model, group_test_data)
            eval_time = time.time() - start_eval_time
            
            group_metrics['evaluation_time'] = eval_time
            group_metrics['test_users_count'] = len(group_test_data['user_id'].unique())
            
            results['user_group_results'][user_group] = group_metrics
            
            print(f"    {user_group}: P@10={group_metrics['precision_at_k']:.4f}, "
                  f"R@10={group_metrics['recall_at_k']:.4f}, Users={group_metrics['test_users_count']}")
        
        return results
    
    def _cleanup_memory_after_model(self):
        """Cleanup memory after training a model."""
        import gc
        
        # Try to cleanup neural models if they exist
        try:
            from approaches.neural import cleanup_model
            # This will be called for any model cleanup
            gc.collect()
        except ImportError:
            pass
        
        # Explicit garbage collection
        gc.collect()
    
    def _get_current_memory(self) -> float:
        """Get the current memory usage."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def check_existing_results(self, experiment_name: str, exp_config: Dict[str, Any], 
                             output_dir: str = "results") -> bool:
        """
        Check if results already exist for this experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            exp_config: Experiment configuration
            output_dir: Results directory to check
            
        Returns:
            True if matching results exist, False otherwise
        """
        experiment_dir = os.path.join(output_dir, experiment_name)
        
        if not os.path.exists(experiment_dir):
            return False
        
        # Look for existing experiment result files
        import glob
        result_files = glob.glob(os.path.join(experiment_dir, "experiment_results_*.json"))
        
        if not result_files:
            return False
        
        # Get current experiment configuration signature
        current_config = {
            'models': sorted(exp_config["models"]),  # Sort for consistent comparison
            'user_groups': sorted(exp_config["user_groups"]),  # Use config user_groups (list)
            'data_sample_size': self.config.get('SAMPLE_SIZE'),
            'positive_threshold': self.config.get('POSITIVE_RATING_THRESHOLD'),
            'data_path': self.config.get('DATA_PATH')
        }
        
        # Check each existing result file
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    existing_results = json.load(f)
                
                existing_config = existing_results.get('config', {})
                
                # Compare key configuration parameters
                # Handle user_groups comparison (can be list or dict in existing results)
                existing_user_groups = existing_config.get('user_groups', [])
                if isinstance(existing_user_groups, dict):
                    # Extract group names from dict format
                    existing_group_names = list(existing_user_groups.keys())
                else:
                    # Already in list format
                    existing_group_names = existing_user_groups
                
                # Compare only the essential parameters that matter for experiment identity
                models_match = sorted(existing_config.get('models', [])) == current_config['models']
                groups_match = sorted(existing_group_names) == sorted(current_config['user_groups'])
                sample_match = existing_config.get('data_sample_size') == current_config['data_sample_size']
                
                # Handle legacy results where threshold might be None/missing
                existing_threshold = existing_config.get('positive_threshold')
                current_threshold = current_config['positive_threshold']
                threshold_match = (
                    existing_threshold == current_threshold or  # Exact match
                    (existing_threshold is None and current_threshold == 3.5)  # Legacy with default
                )
                
                config_matches = models_match and groups_match and sample_match and threshold_match
                
                if config_matches:
                    # Check if all expected models completed successfully
                    model_results = existing_results.get('model_results', {})
                    expected_models = current_config['models']
                    completed_models = set(model_results.keys())
                    failed_models = [name for name, result in model_results.items() 
                                   if 'error' in result]
                    missing_models = set(expected_models) - completed_models
                    
                    if not failed_models and not missing_models:
                        print(f"Found existing complete results for {experiment_name}")
                        print(f"    File: {os.path.basename(result_file)}")
                        print(f"    Models: {len(model_results)}/{len(expected_models)} successful")
                        print(f"    Completed: {sorted(completed_models)}")
                        print(f"    Config: sample_size={current_config['data_sample_size']}, "
                              f"threshold={current_config['positive_threshold']}")
                        return True
                    else:
                        print(f"Found incomplete results for {experiment_name}")
                        if failed_models:
                            print(f"    Failed models: {failed_models}")
                        if missing_models:
                            print(f"    Missing models: {sorted(missing_models)}")
                        print(f"    Will re-run to complete missing results")
                        
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error reading {result_file}: {e}")
                continue
        
        return False

    def run_experiment_suite(self, experiment_name: str, output_dir: str = "results", 
                           force_rerun: bool = False, custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a complete experiment suite.
        
        Args:
            experiment_name: Name of experiment configuration
            output_dir: Output directory for results
            force_rerun: Force re-run even if results exist
            
        Returns:
            Complete experiment results
        """
        print(f"Starting experiment suite: {experiment_name}")
        
        # Get experiment configuration
        if experiment_name in config.EXPERIMENT_CONFIGS:
            exp_config = config.EXPERIMENT_CONFIGS[experiment_name]
            models = exp_config["models"]
            user_groups = exp_config["user_groups"]
        elif custom_config:
            exp_config = custom_config
            models = exp_config["models"]
            user_groups = exp_config["user_groups"]
        else:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        
        # Check for existing results and determine which models to run
        existing_results = {}
        models_to_run = models.copy()
        
        if not force_rerun:
            # Try to load existing results
            experiment_output_dir = os.path.join(output_dir, experiment_name)
            import glob
            result_files = glob.glob(os.path.join(experiment_output_dir, "experiment_results_*.json"))
            
            if result_files:
                try:
                    with open(result_files[0], 'r') as f:
                        existing_results = json.load(f)
                    
                    # Check which models are already completed
                    completed_models = set(existing_results.get('model_results', {}).keys())
                    failed_models = [name for name, result in existing_results.get('model_results', {}).items() 
                                   if 'error' in result]
                    
                    # Only run models that are missing or failed
                    models_to_run = [m for m in models if m not in completed_models or m in failed_models]
                    
                    if not models_to_run:
                        print(f"All models already completed for {experiment_name}")
                        return existing_results
                    else:
                        print(f"Resuming {experiment_name} - running remaining models: {models_to_run}")
                        print(f"Already completed: {sorted(completed_models - set(failed_models))}")
                except Exception as e:
                    print(f"Error loading existing results: {e}")
                    models_to_run = models  # Run all models if loading fails
        
        # Prepare data using global data manager
        train_data, test_data, user_groups = self.prepare_data()
        
        # Initialize results (load existing or create new)
        if existing_results:
            experiment_results = existing_results
        else:
            experiment_results = {
                'experiment_name': experiment_name,
                'experiment_description': exp_config["description"],
                'start_time': self.experiment_start_time.isoformat(),
                'config': {
                    'models': models,
                    'user_groups': exp_config["user_groups"],  # Use config user_groups (list)
                    'data_sample_size': self.config.get('SAMPLE_SIZE'),
                    'positive_threshold': self.config.get('POSITIVE_RATING_THRESHOLD'),
                    'data_path': self.config.get('DATA_PATH')
                },
                'model_results': {}
            }
        
        for model_name in models_to_run:
            try:
                print(f"\n=== Starting {model_name} (Memory: {self._get_current_memory():.1f}MB) ===")
                
                model_results = self.run_single_experiment(
                    model_name, train_data, test_data, user_groups
                )
                experiment_results['model_results'][model_name] = model_results
                
                # Save results after each model (for long-running experiments)
                if len(experiment_results['model_results']) > 0:
                    current_results = experiment_results.copy()
                    current_results['models_completed'] = list(experiment_results['model_results'].keys())
                    # Save to the same experiment folder (no incremental suffix)
                    self.save_results(current_results, experiment_name)
                    print(f"✓ Results saved after {model_name}")
                
                # Explicit memory cleanup between models
                print(f"Cleaning up memory after {model_name}...")
                self._cleanup_memory_after_model()
                print(f"Memory after cleanup: {self._get_current_memory():.1f}MB")
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                experiment_results['model_results'][model_name] = {
                    'error': str(e),
                    'model_name': model_name
                }
                # Cleanup even after errors
                self._cleanup_memory_after_model()
        
        # Add summary statistics
        experiment_results['summary'] = self.generate_experiment_summary(experiment_results)
        experiment_results['end_time'] = datetime.now().isoformat()
        
        # Save main results file
        self.save_results(experiment_results, experiment_name)
        
        return experiment_results
    
    def generate_experiment_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for experiment results.
        
        Args:
            results: Complete experiment results
            
        Returns:
            Summary statistics
        """
        summary = {
            'total_models': len(results['model_results']),
            'successful_models': 0,
            'failed_models': 0,
            'best_models_by_metric': {},
            'approach_comparison': {}
        }
        
        # Collect all successful results
        successful_results = []
        for model_name, model_result in results['model_results'].items():
            if 'error' in model_result:
                summary['failed_models'] += 1
            else:
                summary['successful_models'] += 1
                
                # Add to results for analysis
                for user_group, metrics in model_result.get('user_group_results', {}).items():
                    successful_results.append({
                        'model': model_name,
                        'user_group': user_group,
                        'approach': metrics.get('approach_type'),
                        'complexity': metrics.get('complexity_level'),
                        'precision': metrics.get('precision_at_k', 0),
                        'recall': metrics.get('recall_at_k', 0),
                        'ndcg': metrics.get('ndcg_at_k', 0)
                    })
        
        # Find best models by metric
        if successful_results:
            df_results = pd.DataFrame(successful_results)
            
            for metric in ['precision', 'recall', 'ndcg']:
                best_idx = df_results[metric].idxmax()
                best_result = df_results.iloc[best_idx]
                summary['best_models_by_metric'][metric] = {
                    'model': best_result['model'],
                    'user_group': best_result['user_group'],
                    'value': best_result[metric]
                }
            
            # Approach comparison
            if 'approach' in df_results.columns:
                approach_stats = df_results.groupby('approach').agg({
                    'precision': ['mean', 'std', 'count'],
                    'recall': ['mean', 'std', 'count'],
                    'ndcg': ['mean', 'std', 'count']
                }).round(4)
                
                summary['approach_comparison'] = approach_stats.to_dict()
        
        return summary
    
    def save_results(self, results: Dict[str, Any], experiment_name: str) -> None:
        """
        Save experiment results to files, using the path from config.
        """
        # Use the path from the config file
        output_dir = config.RESULTS_PATH
        
        # Create the specific output directory for the experiment suite
        experiment_output_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(experiment_output_dir, exist_ok=True)
        
        sample_size = self.config.get('SAMPLE_SIZE')
        sample_suffix = f"_sample_{sample_size}" if sample_size else "_full"
        
        # Save to main results file (no incremental suffix)
        results_file = os.path.join(experiment_output_dir, f"experiment_results{sample_suffix}.json")
        
        # Create a serializable copy
        def convert_keys(obj):
            if isinstance(obj, dict):
                return {str(k): convert_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys(v) for v in obj]
            else:
                return obj
        
        serializable_results = convert_keys(results)
        json.dump(serializable_results, open(results_file, 'w'), indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
        
        # Save individual model results for compatibility with sample size info
        
        for model_name, model_result in results['model_results'].items():
            if 'error' not in model_result:
                model_dir = os.path.join(experiment_output_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)
                
                # Save evaluation metrics with sample size info
                metrics_file = os.path.join(model_dir, f"evaluation_metrics{sample_suffix}.json")
                with open(metrics_file, 'w') as f:
                    json.dump(model_result, f, indent=2, default=str)
        
        print(f"Individual model results saved to: {experiment_output_dir}/[model_name]/evaluation_metrics{sample_suffix}.json")

    def run_double_descent_experiment(self, output_dir: str = "results", force_rerun: bool = False):
        """
        Run experiment to test the double descent hypothesis by varying model complexity.
        """
        dd_config = config.DOUBLE_DESCENT_CONFIG
        if not dd_config.get("enabled"):
            print("Double descent experiment is disabled in config.")
            return

        print("Starting Double Descent Experiment...")
        
        output_path = os.path.join(output_dir, dd_config["output_dir"])
        os.makedirs(output_path, exist_ok=True)
        
        # Check if double descent experiment is already complete (unless force_rerun)
        if not force_rerun:
            results_file = os.path.join(output_path, "double_descent_results.json")
            if os.path.exists(results_file):
                print(f"Found existing double descent results: {results_file}")
                print("Double descent experiment already completed. Use --force_rerun to run again.")
                return
            
            # Check for partial results and resume if possible
            completed_models = set()
            for model_name in dd_config["models"].keys():
                model_file = os.path.join(output_path, f"double_descent_{model_name}.json")
                
                if os.path.exists(model_file):
                    # Load file to see how many values were completed
                    try:
                        with open(model_file, 'r') as f:
                            model_data = json.load(f)
                            if model_name in model_data:
                                completed_values = len(model_data[model_name])
                                total_values = len(dd_config["models"][model_name]["values"])
                                if completed_values >= total_values:
                                    completed_models.add(model_name)
                                    print(f"Found completed results for {model_name} ({completed_values}/{total_values})")
                                else:
                                    print(f"Found partial results for {model_name} ({completed_values}/{total_values})")
                    except:
                        pass
            
            if completed_models:
                print(f"Found {len(completed_models)} completed models: {sorted(completed_models)}")
                print("Double descent experiment partially completed. Use --force_rerun to restart from scratch.")
                return
        
        full_results = {}

        # Prepare data once for all runs
        train_data, test_data, _ = self.prepare_data()

        for model_name, experiment_params in dd_config["models"].items():
            param_name = experiment_params["param_to_vary"]
            param_values = experiment_params["values"]
            
            print(f"Testing model: {model_name} by varying '{param_name}'")
            
            # Load existing results if available
            model_results = []
            model_file = os.path.join(output_path, f"double_descent_{model_name}.json")
            if os.path.exists(model_file) and not force_rerun:
                try:
                    with open(model_file, 'r') as f:
                        existing_data = json.load(f)
                        if model_name in existing_data:
                            model_results = existing_data[model_name]
                            print(f"  Loaded {len(model_results)} existing results for {model_name}")
                except:
                    pass
            
            # Find which values still need to be tested
            completed_values = {result["complexity_value"] for result in model_results}
            remaining_values = [v for v in param_values if v not in completed_values]
            
            if not remaining_values:
                print(f"  All {len(param_values)} values already completed for {model_name}")
                full_results[model_name] = model_results
                continue
            
            print(f"  Testing {len(remaining_values)} remaining values: {remaining_values}")

            initial_count = len(model_results)
            for i, value in enumerate(remaining_values):
                current_count = initial_count + i + 1
                print(f"  -> Testing with {param_name} = {value} ({current_count}/{len(param_values)})")
                print(f"     Memory before: {self._get_current_memory():.1f}MB")
                
                try:
                    # Load the base model
                    model = self.load_model_class(model_name)
                    
                    # Override the complexity parameter
                    if hasattr(model, param_name):
                        setattr(model, param_name, value)
                    elif hasattr(model, 'config') and param_name in model.config: # For neural models
                        model.config[param_name] = value
                    else:
                        print(f"Could not set parameter '{param_name}' for model {model_name}. Skipping value {value}.")
                        continue

                    # Train the model
                    model.fit(train_data)
                    
                    # Evaluate on the full test set
                    metrics = self.evaluate_model(model, test_data, k=config.DEFAULT_K)
                    
                    # Store relevant results
                    result_point = {
                        "complexity_param": param_name,
                        "complexity_value": value,
                        "precision_at_k": metrics.get("precision_at_k"),
                        "recall_at_k": metrics.get("recall_at_k"),
                        "ndcg_at_k": metrics.get("ndcg_at_k"),
                        "rmse": metrics.get("rmse")
                    }
                    model_results.append(result_point)
                    rmse_str = f"{result_point['rmse']:.4f}" if result_point['rmse'] is not None else "N/A"
                    print(f"    -> Precision@10: {result_point['precision_at_k']:.4f}, RMSE: {rmse_str}")
                    
                except Exception as e:
                    print(f"    -> Error with {param_name} = {value}: {e}")
                    result_point = {
                        "complexity_param": param_name,
                        "complexity_value": value,
                        "error": str(e)
                    }
                    model_results.append(result_point)
                
                # Memory cleanup after each complexity variation
                print(f"     Cleaning up memory after {param_name} = {value}...")
                self._cleanup_memory_after_model()
                print(f"     Memory after cleanup: {self._get_current_memory():.1f}MB")
                
                # Save incremental results after each variation
                model_results_file = os.path.join(output_path, f"double_descent_{model_name}.json")
                with open(model_results_file, 'w') as f:
                    json.dump({model_name: model_results}, f, indent=2)

            full_results[model_name] = model_results
            print(f"  ✓ Results saved for {model_name}")
        
        # Save complete results to a JSON file
        results_file = os.path.join(output_path, "double_descent_results.json")
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
            
        print(f"Double descent experiment finished. Results saved to {results_file}")


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description="Run TTI Recommendation System Experiments")
    
    # Group for standard experiments
    std_group = parser.add_argument_group('Standard Experiments')
    std_group.add_argument("--experiment", type=str, help="Name of experiment suite to run from config")
    std_group.add_argument("--approach", type=str, help="Run all models of a specific approach (e.g., content)")
    std_group.add_argument("--complexity", type=str, help="Run all models of a specific complexity (low, medium, high)")
    std_group.add_argument("--models", type=str, help="Comma-separated list of specific models to run")

    # Group for double descent experiment
    dd_group = parser.add_argument_group('Double Descent Experiment')
    dd_group.add_argument("--skip_double_descent", action="store_true", help="Skip the double descent experiment")

    # General options
    opt_group = parser.add_argument_group('General Options')
    opt_group.add_argument("--force_rerun", action="store_true", help="Force rerun experiments even if results exist")
    opt_group.add_argument("--sample_size", type=int, help="Override sample size from config for quick tests")
    
    args = parser.parse_args()
    
    # Initialize runner with overrides from args
    runner = ExperimentRunner(args_overrides=vars(args))
    
    # --- Execution Logic ---

    # Determine models for standard runs
    models_to_run = []
    experiment_name = "custom_run"
    
    if args.models:
        models_to_run = args.models.split(',')
        experiment_name = f"custom_{'_'.join(models_to_run)}"
    elif args.complexity:
        models_to_run = config.COMPLEXITY_LEVELS.get(args.complexity, [])
        experiment_name = f"complexity_{args.complexity}"
    elif args.approach:
        models_to_run = config.APPROACH_TYPES.get(args.approach, [])
        experiment_name = f"approach_{args.approach}"
    
    # Run a specific experiment suite from config
    if args.experiment:
        print(f"Running experiment suite: {args.experiment}")
        runner.run_experiment_suite(args.experiment, output_dir=config.RESULTS_PATH, force_rerun=args.force_rerun)
        
        # Run double descent after if not skipped
        if not args.skip_double_descent:
            print(f"\n{'='*20} Running: Double Descent Experiment {'='*20}")
            runner.run_double_descent_experiment(output_dir=config.RESULTS_PATH, force_rerun=args.force_rerun)
        return
    
    # Run custom set of models if specified
    if models_to_run:
        print(f"Running custom experiment with models: {models_to_run}")
        exp_config = {
            'models': models_to_run,
            'user_groups': list(config.USER_GROUP_THRESHOLDS.keys()) + ['full'],
            'description': f"Custom run for models: {', '.join(models_to_run)}"
        }
        
        # Check for existing results first
        if not args.force_rerun and runner.check_existing_results(experiment_name, exp_config, config.RESULTS_PATH):
            # Still run double descent if not skipped
            if not args.skip_double_descent:
                print(f"\n{'='*20} Running: Double Descent Experiment {'='*20}")
                runner.run_double_descent_experiment(output_dir=config.RESULTS_PATH, force_rerun=args.force_rerun)
            return # Exit if results exist and not forcing rerun

        results = runner.run_experiment_suite(experiment_name, config.RESULTS_PATH, args.force_rerun, custom_config=exp_config)
        
        # Run double descent after if not skipped
        if not args.skip_double_descent:
            print(f"\n{'='*20} Running: Double Descent Experiment {'='*20}")
            runner.run_double_descent_experiment(output_dir=config.RESULTS_PATH, force_rerun=args.force_rerun)
        return

    # Default behavior: run all enabled experiments from config
    print("Running all enabled experiments from config.py...")
    for exp_name in config.get_enabled_experiments():
        print(f"\n{'='*20} Running: {exp_name} {'='*20}")
        runner.run_experiment_suite(exp_name, output_dir=config.RESULTS_PATH, force_rerun=args.force_rerun)
    
    # Run double descent experiment last (if not skipped)
    if not args.skip_double_descent:
        print(f"\n{'='*20} Running: Double Descent Experiment {'='*20}")
        runner.run_double_descent_experiment(output_dir=config.RESULTS_PATH, force_rerun=args.force_rerun)

if __name__ == "__main__":
    main() 