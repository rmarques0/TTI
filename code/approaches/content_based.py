import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.base_recommender import BaseRecommender
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import shap

class ContentBasedLogistic(BaseRecommender):
    """
    Content-based recommendation using Logistic Regression.
    Low complexity model.
    """
    
    def __init__(self):
        super().__init__("content_logistic", "low", "content")
        self.scaler = StandardScaler()
        
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit logistic regression on content features."""
        # Combine user and item features for a true content-based model
        content_features = [col for col in train_data.columns if any(prefix in col for prefix in [
            # User features
            'user_avg_rating', 'user_rating_count', 'gender_encoded',
            'occupation_encoded', 'age_group_encoded',
            # Item features
            'genre_', 'avg_rating', 'rating_count'
        ])]
        
        X = train_data[content_features].fillna(0)
        y = train_data['is_positive']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_scaled, y)
        
        self.feature_names = content_features
        self.is_fitted = True
        
        # Calculate feature importance for explainability using SHAP
        # LinearExplainer is optimized for linear models
        explainer = shap.LinearExplainer(self.model, X_scaled)
        shap_values = explainer.shap_values(X_scaled)
        
        # Summarize the absolute SHAP values across all samples
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        self.explainability_data = {
            'feature_importance': dict(zip(self.feature_names, mean_abs_shap.tolist())),
            'model_type': 'logistic_regression',
            'approach': 'content_based'
        }
    
    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations using content similarity."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use the same features as training
        X_test = test_data[self.feature_names].fillna(0)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Create recommendations by user
        recommendations = {}
        test_data_with_probs = test_data.copy()
        test_data_with_probs['pred_prob'] = probabilities
        
        for user_id in test_data['user_id'].unique():
            user_items = test_data_with_probs[test_data_with_probs['user_id'] == user_id]
            top_k = user_items.nlargest(k, 'pred_prob')
            recommendations[user_id] = top_k['movie_id'].tolist()
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from logistic regression coefficients."""
        if not self.is_fitted:
            return {}
        return self.explainability_data['feature_importance']


class ContentBasedKNN(BaseRecommender):
    """
    Content-based recommendation using K-Nearest Neighbors.
    Low complexity model.
    """
    
    def __init__(self, n_neighbors: int = 50):
        super().__init__("content_knn", "low", "content")
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit KNN on content features."""
        # Combine user and item features for a true content-based model
        content_features = [col for col in train_data.columns if any(prefix in col for prefix in [
            # User features
            'user_avg_rating', 'user_rating_count', 'gender_encoded',
            'occupation_encoded', 'age_group_encoded',
            # Item features
            'genre_', 'avg_rating', 'rating_count'
        ])]
        
        X = train_data[content_features].fillna(0)
        y = train_data['is_positive']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(X_scaled, y)
        
        self.feature_names = content_features
        self.is_fitted = True
        
        # Use SHAP's KernelExplainer for KNN (model-agnostic)
        # Note: This can be slow, so we use a small sample
        background_data = shap.sample(X_scaled, 100)
        explainer = shap.KernelExplainer(self.model.predict_proba, background_data)
        
        # Explain a small sample of instances
        shap_values = explainer.shap_values(X_scaled[:200, :])
        
        # We are interested in the explanations for the positive class
        mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)

        self.explainability_data = {
            'feature_importance': dict(zip(self.feature_names, mean_abs_shap.tolist())),
            'model_type': 'knn',
            'approach': 'content_based',
            'n_neighbors': self.n_neighbors
        }
    
    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations using KNN predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use the same features as training
        X_test = test_data[self.feature_names].fillna(0)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Create recommendations by user
        recommendations = {}
        test_data_with_probs = test_data.copy()
        test_data_with_probs['pred_prob'] = probabilities
        
        for user_id in test_data['user_id'].unique():
            user_items = test_data_with_probs[test_data_with_probs['user_id'] == user_id]
            top_k = user_items.nlargest(k, 'pred_prob')
            recommendations[user_id] = top_k['movie_id'].tolist()
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from variance analysis."""
        if not self.is_fitted:
            return {}
        return self.explainability_data.get('feature_importance', {})


class ContentBasedRandomForest(BaseRecommender):
    """
    Content-based recommendation using Random Forest.
    Medium complexity model.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("content_rf", "medium", "content")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit Random Forest on content features."""
        # Combine user and item features for a true content-based model
        content_features = [col for col in train_data.columns if any(prefix in col for prefix in [
            # User features
            'user_avg_rating', 'user_rating_count', 'gender_encoded',
            'occupation_encoded', 'age_group_encoded',
            # Item features
            'genre_', 'avg_rating', 'rating_count'
        ])]
        
        X = train_data[content_features].fillna(0)
        y = train_data['is_positive']
        
        # Fit model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            random_state=42,
            max_depth=self.max_depth,
            min_samples_split=10
        )
        self.model.fit(X, y)
        
        self.feature_names = content_features
        self.is_fitted = True
        
        # Calculate feature importance for explainability using SHAP
        # Using a subset of data for performance
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X.sample(min(1000, X.shape[0]), random_state=42))
        
        # For classification, shap_values is a list of arrays (one for each class)
        # We'll use the values for the positive class (class 1)
        shap_values_class1 = shap_values[1]
        
        # Summarize the absolute SHAP values across all samples
        mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
        
        self.explainability_data = {
            'feature_importance': dict(zip(self.feature_names, mean_abs_shap.tolist())),
            'model_type': 'random_forest',
            'approach': 'content_based',
            'n_estimators': self.n_estimators
        }
    
    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations using Random Forest predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use the same features as training
        X_test = test_data[self.feature_names].fillna(0)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X_test)[:, 1]
        
        # Create recommendations by user
        recommendations = {}
        test_data_with_probs = test_data.copy()
        test_data_with_probs['pred_prob'] = probabilities
        
        for user_id in test_data['user_id'].unique():
            user_items = test_data_with_probs[test_data_with_probs['user_id'] == user_id]
            top_k = user_items.nlargest(k, 'pred_prob')
            recommendations[user_id] = top_k['movie_id'].tolist()
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest."""
        if not self.is_fitted:
            return {}
        return self.explainability_data['feature_importance']


class ContentBasedGradientBoosting(BaseRecommender):
    """
    Content-based recommendation using Gradient Boosting.
    Medium complexity model.
    """
    
    def __init__(self, n_estimators: int = 100):
        super().__init__("content_gb", "medium", "content")
        self.n_estimators = n_estimators
        
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit Gradient Boosting on content features."""
        # Combine user and item features for a true content-based model
        content_features = [col for col in train_data.columns if any(prefix in col for prefix in [
            # User features
            'user_avg_rating', 'user_rating_count', 'gender_encoded',
            'occupation_encoded', 'age_group_encoded',
            # Item features
            'genre_', 'avg_rating', 'rating_count'
        ])]
        
        X = train_data[content_features].fillna(0)
        y = train_data['is_positive']
        
        # Fit model
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            random_state=42,
            max_depth=6,
            learning_rate=0.1
        )
        self.model.fit(X, y)
        
        self.feature_names = content_features
        self.is_fitted = True
        
        # Calculate feature importance using SHAP
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X.sample(min(1000, X.shape[0]), random_state=42))
        
        # For binary classification, shap_values is a single array of shape (n_samples, n_features)
        # Summarize the absolute SHAP values across all samples
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        self.explainability_data = {
            'feature_importance': dict(zip(self.feature_names, mean_abs_shap.tolist())),
            'model_type': 'gradient_boosting',
            'approach': 'content_based',
            'n_estimators': self.n_estimators
        }
    
    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations using Gradient Boosting predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use the same features as training
        X_test = test_data[self.feature_names].fillna(0)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X_test)[:, 1]
        
        # Create recommendations by user
        recommendations = {}
        test_data_with_probs = test_data.copy()
        test_data_with_probs['pred_prob'] = probabilities
        
        for user_id in test_data['user_id'].unique():
            user_items = test_data_with_probs[test_data_with_probs['user_id'] == user_id]
            top_k = user_items.nlargest(k, 'pred_prob')
            recommendations[user_id] = top_k['movie_id'].tolist()
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Gradient Boosting."""
        if not self.is_fitted:
            return {}
        return self.explainability_data['feature_importance']


class ContentBasedSimilarity(BaseRecommender):
    """
    Content-based recommendation using item-item similarity.
    Low complexity model using cosine similarity.
    """
    
    def __init__(self):
        super().__init__("content_similarity", "low", "content")
        self.item_features = None
        self.similarity_matrix = None
        
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit content similarity model."""
        # Get content features for items
        item_features = [col for col in train_data.columns if any(prefix in col for prefix in [
            'genre_', 'avg_rating', 'rating_count', 'popularity_score', 'year'
        ])]
        
        # Create item feature matrix
        item_data = train_data.groupby('movie_id')[item_features].first().fillna(0)
        
        # Calculate item-item similarity
        self.similarity_matrix = cosine_similarity(item_data.values)
        self.item_features = item_data
        self.feature_names = item_features
        self.is_fitted = True
        
        # For explainability, track feature variance
        feature_variance = np.var(item_data.values, axis=0)
        self.explainability_data = {
            'feature_variance': dict(zip(self.feature_names, feature_variance.tolist())),
            'model_type': 'content_similarity',
            'approach': 'content_based',
            'similarity_method': 'cosine'
        }
    
    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations using content similarity."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        recommendations = {}
        available_items = list(self.item_features.index)
        
        for user_id in test_data['user_id'].unique():
            # Since this is content-based similarity, we need to make recommendations
            # based on the available items and their similarities
            
            # For simplicity, let's recommend the top-k most similar items
            # In a real system, we'd use user's historical preferences
            
            if len(available_items) == 0:
                recommendations[user_id] = []
                continue
                
            # Use the first item as reference and find similar items
            # In practice, you'd use user's past preferences from training data
            if len(available_items) >= k:
                # Get items with highest average similarity to all other items
                avg_similarities = np.mean(self.similarity_matrix, axis=0)
                top_indices = np.argsort(avg_similarities)[-k:][::-1]
                recommendations[user_id] = [available_items[idx] for idx in top_indices]
            else:
                recommendations[user_id] = available_items[:k]
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from variance analysis."""
        if not self.is_fitted:
            return {}
        return self.explainability_data['feature_variance'] 