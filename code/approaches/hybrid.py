"""
Hybrid recommendation models.
Combines multiple approaches for better performance.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from core.base_recommender import BaseRecommender
from .content_based import ContentBasedLogistic, ContentBasedRandomForest
from .collaborative_filtering import CollaborativeSVD, CollaborativeUserBased

logger = logging.getLogger(__name__)

class HybridWeighted(BaseRecommender):
    """Weighted hybrid combining content-based and collaborative filtering."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize weighted hybrid model."""
        super().__init__("hybrid_weighted", "medium", "hybrid")
        self.content_weight = config.get("content_weight", 0.5)
        self.collaborative_weight = config.get("collaborative_weight", 0.5)
        
        # Initialize sub-models
        self.content_model = ContentBasedLogistic()
        self.collaborative_model = CollaborativeSVD(n_components=50)
        
        self.content_similarities = None
        self.collaborative_similarities = None
        self.combined_similarities = None
        
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit both content and collaborative models."""
        logger.info("Training weighted hybrid model...")
        
        # Fit content-based model
        self.content_model.fit(train_data)
        
        # Fit collaborative model
        self.collaborative_model.fit(train_data)
        
        self.is_fitted = True
        logger.info(f"Fitted hybrid model (content: {self.content_weight}, collaborative: {self.collaborative_weight})")
    
    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate hybrid recommendations using weighted combination."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from both models
        content_recs = self.content_model.predict(test_data, k)
        collaborative_recs = self.collaborative_model.predict(test_data, k)
        
        # Combine recommendations with weights
        recommendations = {}
        all_users = set(content_recs.keys()) | set(collaborative_recs.keys())
        
        for user_id in all_users:
            content_items = content_recs.get(user_id, [])
            collaborative_items = collaborative_recs.get(user_id, [])
            
            # Score items based on their ranking in each approach
            item_scores = {}
            
            # Content-based scores (higher rank = higher score)
            for rank, item_id in enumerate(content_items):
                score = (len(content_items) - rank) * self.content_weight
                item_scores[item_id] = item_scores.get(item_id, 0) + score
            
            # Collaborative scores
            for rank, item_id in enumerate(collaborative_items):
                score = (len(collaborative_items) - rank) * self.collaborative_weight
                item_scores[item_id] = item_scores.get(item_id, 0) + score
            
            # Get top-k recommendations
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations[user_id] = [item_id for item_id, _ in sorted_items[:k]]
        
        return recommendations
    
    def get_recommendations(self, item_idx: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get hybrid recommendations using weighted similarities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        if item_idx >= len(self.combined_similarities):
            raise ValueError(f"Item index {item_idx} out of range")
            
        similarities = self.combined_similarities[item_idx]
        similar_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in similar_indices:
            if len(recommendations) >= n_recommendations:
                break
            if idx != item_idx:
                recommendations.append((idx, similarities[idx]))
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from hybrid model components."""
        if not self.is_fitted:
            return {}
        
        importance = {}
        
        # Get importance from content model
        if hasattr(self.content_model, 'get_feature_importance'):
            content_importance = self.content_model.get_feature_importance()
            for feature, score in content_importance.items():
                importance[f"content_{feature}"] = score * self.content_weight
        
        # Get importance from collaborative model  
        if hasattr(self.collaborative_model, 'get_feature_importance'):
            collab_importance = self.collaborative_model.get_feature_importance()
            for feature, score in collab_importance.items():
                importance[f"collaborative_{feature}"] = score * self.collaborative_weight
        
        # Add hybrid-specific importance
        importance["hybrid_content_weight"] = self.content_weight
        importance["hybrid_collaborative_weight"] = self.collaborative_weight
        
        return importance

class HybridStacking(BaseRecommender):
    """Stacking hybrid using meta-learner to combine models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize stacking hybrid model."""
        super().__init__("hybrid_stacking", "high", "hybrid")
        self.meta_learner_type = config.get("meta_learner", "logistic")
        
        # Initialize base models
        self.base_models = [
            ContentBasedLogistic(),
            ContentBasedRandomForest(50),
            CollaborativeSVD(n_components=50),
            CollaborativeUserBased(n_neighbors=30)
        ]
        
        # Initialize meta-learner
        if self.meta_learner_type == "logistic":
            self.meta_learner = LogisticRegression(random_state=42)
        else:
            self.meta_learner = RandomForestRegressor(n_estimators=50, random_state=42)
        
        self.base_predictions = None
        self.final_similarities = None
        
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit base models and meta-learner."""
        logger.info("Training stacking hybrid model...")
        
        # Train base models
        self.trained_models = []
        for i, model in enumerate(self.base_models):
            try:
                model.fit(train_data)
                self.trained_models.append(model)
                logger.info(f"Trained base model {i+1}/{len(self.base_models)}")
            except Exception as e:
                logger.warning(f"Failed to train base model {i}: {e}")
        
        self.is_fitted = True
        logger.info(f"Fitted stacking hybrid with {len(self.trained_models)} base models")
    
    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Predict using stacked models."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Get predictions from all trained models
        all_predictions = []
        for model in self.trained_models:
            try:
                pred = model.predict(test_data, k)
                all_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Base model prediction failed: {e}")
        
        if not all_predictions:
            # No successful predictions
            return {user_id: [] for user_id in test_data['user_id'].unique()}
        
        # Combine predictions using voting/averaging
        recommendations = {}
        all_users = set()
        for pred in all_predictions:
            all_users.update(pred.keys())
        
        for user_id in all_users:
            item_scores = {}
            
            # Score items based on their ranking in each model
            for pred in all_predictions:
                user_items = pred.get(user_id, [])
                for rank, item_id in enumerate(user_items):
                    score = len(user_items) - rank  # Higher rank = higher score
                    item_scores[item_id] = item_scores.get(item_id, 0) + score
            
            # Get top-k recommendations
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations[user_id] = [item_id for item_id, _ in sorted_items[:k]]
        
        return recommendations
    
    def get_recommendations(self, item_idx: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get stacked recommendations."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        if item_idx >= len(self.final_similarities):
            raise ValueError(f"Item index {item_idx} out of range")
            
        similarities = self.final_similarities[item_idx]
        similar_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in similar_indices:
            if len(recommendations) >= n_recommendations:
                break
            if idx != item_idx:
                recommendations.append((idx, similarities[idx]))
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from stacked base models."""
        if not self.is_fitted:
            return {}
        
        importance = {}
        
        # Get importance from each base model
        for i, model in enumerate(self.base_models):
            if hasattr(model, 'get_feature_importance'):
                model_importance = model.get_feature_importance()
                for feature, score in model_importance.items():
                    importance[f"base_model_{i}_{feature}"] = score / len(self.base_models)
        
        # Add stacking-specific importance
        importance["stacking_num_base_models"] = len(self.base_models)
        importance["stacking_meta_learner"] = 1.0 if self.meta_learner_type == "logistic" else 0.5
        
        return importance

class HybridSwitching(BaseRecommender):
    """Switching hybrid that selects best model based on context."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize switching hybrid model."""
        super().__init__("hybrid_switching", "medium", "hybrid")
        self.threshold = config.get("switching_threshold", 0.1)
        
        # Initialize models for different scenarios
        self.content_model = ContentBasedRandomForest(100)
        self.collaborative_model = CollaborativeSVD(n_components=100)
        
        self.content_similarities = None
        self.collaborative_similarities = None
        
    def fit(self, features: np.ndarray):
        """Fit both models."""
        logger.info("Training switching hybrid model...")
        
        # Fit both models
        self.content_model.fit(features)
        self.content_similarities = self.content_model.similarity_matrix
        
        self.collaborative_model.fit(features)
        self.collaborative_similarities = self.collaborative_model.similarity_matrix
        
        self.is_fitted = True
        logger.info("Fitted switching hybrid model")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict using appropriate model."""
        # Simple switching logic: use collaborative if available, otherwise content
        try:
            return self.collaborative_model.predict(features)
        except:
            return self.content_model.predict(features)
    
    def get_recommendations(self, item_idx: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations using switching logic."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        # Switch based on collaborative performance
        collaborative_scores = self.collaborative_similarities[item_idx]
        max_collab_score = np.max(collaborative_scores)
        
        # If collaborative has good similarities, use it; otherwise use content
        if max_collab_score > self.threshold:
            similarities = collaborative_scores
        else:
            similarities = self.content_similarities[item_idx]
            
        similar_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in similar_indices:
            if len(recommendations) >= n_recommendations:
                break
            if idx != item_idx:
                recommendations.append((idx, similarities[idx]))
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from switching hybrid components."""
        if not self.is_fitted:
            return {}
        
        importance = {}
        
        # Get importance from content model
        if hasattr(self.content_model, 'get_feature_importance'):
            content_importance = self.content_model.get_feature_importance()
            for feature, score in content_importance.items():
                importance[f"content_{feature}"] = score * 0.5
        
        # Get importance from collaborative model  
        if hasattr(self.collaborative_model, 'get_feature_importance'):
            collab_importance = self.collaborative_model.get_feature_importance()
            for feature, score in collab_importance.items():
                importance[f"collaborative_{feature}"] = score * 0.5
        
        # Add switching-specific importance
        importance["switching_threshold"] = self.threshold
        importance["switching_logic"] = 1.0
        
        return importance