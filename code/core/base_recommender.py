from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, ndcg_score
import json
import os

class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation models.
    Defines the common interface and shared functionality.
    """
    
    def __init__(self, name: str, complexity_level: str, approach_type: str):
        """
        Initialize the base recommender.
        
        Args:
            name: Model name (e.g., 'logistic_regression', 'svd', 'ncf')
            complexity_level: 'low', 'medium', 'high'
            approach_type: 'content', 'collaborative', 'hybrid'
        """
        self.name = name
        self.complexity_level = complexity_level
        self.approach_type = approach_type
        self.model = None
        self.is_fitted = False
        self.feature_names = []
        self.explainability_data = {}
        
    @abstractmethod
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        Train the recommendation model.
        
        Args:
            train_data: Training dataset
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """
        Generate top-k recommendations for users.
        
        Args:
            test_data: Test dataset
            k: Number of recommendations per user
            
        Returns:
            Dictionary mapping user_id to list of recommended movie_ids
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores for explainability.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def evaluate(self, predictions: Dict[int, List[int]], ground_truth: Dict[int, List[int]]) -> Dict[str, float]:
        """
        Evaluate model performance using standard metrics.
        
        Args:
            predictions: User -> list of predicted movie_ids
            ground_truth: User -> list of actual liked movie_ids
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'precision_at_k': [],
            'recall_at_k': [],
            'ndcg_at_k': [],
            'coverage': 0.0,
            'diversity': 0.0
        }
        
        all_users = set(predictions.keys()) & set(ground_truth.keys())
        
        for user_id in all_users:
            pred_items = predictions.get(user_id, [])
            true_items = ground_truth.get(user_id, [])
            
            if len(true_items) == 0 or len(pred_items) == 0:
                continue
                
            # Precision@K
            relevant_items = set(pred_items) & set(true_items)
            precision = len(relevant_items) / len(pred_items) if len(pred_items) > 0 else 0
            metrics['precision_at_k'].append(precision)
            
            # Recall@K
            recall = len(relevant_items) / len(true_items) if len(true_items) > 0 else 0
            metrics['recall_at_k'].append(recall)
            
            # NDCG@K - only calculate if we have enough items
            if len(pred_items) > 1:
                y_true = [1 if item in true_items else 0 for item in pred_items]
                y_scores = [1.0 - (i / len(pred_items)) for i in range(len(pred_items))]  # Decreasing scores
                
                if sum(y_true) > 0:  # Only if there are relevant items
                    try:
                        ndcg = ndcg_score([y_true], [y_scores], k=len(pred_items))
                        metrics['ndcg_at_k'].append(ndcg)
                    except ValueError:
                        # Skip NDCG if calculation fails
                        pass
        
        # Calculate coverage and diversity
        all_recommended_items = set()
        for recs in predictions.values():
            all_recommended_items.update(recs)
        
        # Coverage: proportion of items recommended
        all_ground_truth_items = set()
        for items in ground_truth.values():
            all_ground_truth_items.update(items)
        
        metrics['coverage'] = len(all_recommended_items) / len(all_ground_truth_items) if len(all_ground_truth_items) > 0 else 0
        
        # Aggregate metrics
        return {
            'precision_at_k': np.mean(metrics['precision_at_k']) if metrics['precision_at_k'] else 0,
            'recall_at_k': np.mean(metrics['recall_at_k']) if metrics['recall_at_k'] else 0,
            'ndcg_at_k': np.mean(metrics['ndcg_at_k']) if metrics['ndcg_at_k'] else 0,
            'coverage': metrics['coverage'],
            'num_users_evaluated': len(all_users)
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Save model results and explainability data.
        
        Args:
            results: Evaluation results and metrics
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save evaluation metrics
        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save explainability data
        if self.explainability_data:
            explain_path = os.path.join(output_dir, "explanation_samples.json")
            with open(explain_path, 'w') as f:
                json.dump(self.explainability_data, f, indent=2)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.name,
            'complexity_level': self.complexity_level,
            'approach_type': self.approach_type,
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names[:10] if len(self.feature_names) > 10 else self.feature_names
        } 