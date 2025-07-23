import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.base_recommender import BaseRecommender
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
import implicit
import warnings
warnings.filterwarnings('ignore')

class CollaborativeSVD(BaseRecommender):
    """
    Collaborative filtering using Singular Value Decomposition.
    Low complexity model.
    """
    
    def __init__(self, n_components: int = 50):
        super().__init__("collaborative_svd", "low", "collaborative")
        self.n_components = n_components
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit SVD on user-item interaction matrix."""
        # Create user and item mappings
        unique_users = sorted(train_data['user_id'].unique())
        unique_items = sorted(train_data['movie_id'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Create interaction matrix
        n_users = len(unique_users)
        n_items = len(unique_items)
        interaction_matrix = np.zeros((n_users, n_items))
        
        for _, row in train_data.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            item_idx = self.item_mapping[row['movie_id']]
            interaction_matrix[user_idx, item_idx] = row['rating']
        
        # Apply SVD
        self.model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_factors = self.model.fit_transform(interaction_matrix)
        self.item_factors = self.model.components_.T
        
        self.is_fitted = True
        
        # For explainability, store component importance
        explained_variance = self.model.explained_variance_ratio_
        self.explainability_data = {
            'explained_variance_ratio': explained_variance.tolist(),
            'n_components': self.n_components,
            'model_type': 'svd',
            'approach': 'collaborative',
            'total_variance_explained': np.sum(explained_variance)
        }
    
    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations using SVD predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        recommendations = {}
        
        for user_id in test_data['user_id'].unique():
            if user_id not in self.user_mapping:
                recommendations[user_id] = []
                continue
                
            user_idx = self.user_mapping[user_id]
            user_vector = self.user_factors[user_idx]
            
            # Calculate scores for all items
            scores = np.dot(user_vector, self.item_factors.T)
            
            # Get top-k items
            top_k_indices = np.argsort(scores)[::-1][:k]
            top_k_items = [self.reverse_item_mapping[idx] for idx in top_k_indices 
                          if idx in self.reverse_item_mapping]
            
            recommendations[user_id] = top_k_items
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from SVD components."""
        if not self.is_fitted:
            return {}
        
        # Use explained variance ratio as feature importance
        importance = {}
        for i, var_ratio in enumerate(self.explainability_data['explained_variance_ratio']):
            importance[f'component_{i}'] = var_ratio
            
        return importance


class CollaborativeUserBased(BaseRecommender):
    """
    Collaborative filtering using user-based nearest neighbors.
    Low complexity model.
    """
    
    def __init__(self, n_neighbors: int = 50):
        super().__init__("collaborative_user_knn", "low", "collaborative")
        self.n_neighbors = n_neighbors
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.interaction_matrix = None
        
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit user-based collaborative filtering."""
        # Create user and item mappings
        unique_users = sorted(train_data['user_id'].unique())
        unique_items = sorted(train_data['movie_id'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Create interaction matrix
        n_users = len(unique_users)
        n_items = len(unique_items)
        self.interaction_matrix = np.zeros((n_users, n_items))
        
        for _, row in train_data.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            item_idx = self.item_mapping[row['movie_id']]
            self.interaction_matrix[user_idx, item_idx] = row['rating']
        
        # Fit KNN model
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')
        self.model.fit(self.interaction_matrix)
        
        self.is_fitted = True
        
        # For explainability, calculate user diversity
        user_similarities = []
        for i in range(min(100, n_users)):  # Sample for efficiency
            distances, indices = self.model.kneighbors([self.interaction_matrix[i]])
            avg_similarity = 1 - np.mean(distances[0])
            user_similarities.append(avg_similarity)
        
        self.explainability_data = {
            'avg_user_similarity': np.mean(user_similarities),
            'n_neighbors': self.n_neighbors,
            'model_type': 'user_based_knn',
            'approach': 'collaborative'
        }
    
    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations using user-based CF."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        recommendations = {}
        
        for user_id in test_data['user_id'].unique():
            if user_id not in self.user_mapping:
                recommendations[user_id] = []
                continue
                
            user_idx = self.user_mapping[user_id]
            user_vector = self.interaction_matrix[user_idx].reshape(1, -1)
            
            # Find similar users
            distances, neighbor_indices = self.model.kneighbors(user_vector)
            neighbor_indices = neighbor_indices[0]
            
            # Calculate item scores based on similar users
            item_scores = {}
            for neighbor_idx in neighbor_indices:
                neighbor_ratings = self.interaction_matrix[neighbor_idx]
                similarity = 1 - distances[0][list(neighbor_indices).index(neighbor_idx)]
                
                for item_idx, rating in enumerate(neighbor_ratings):
                    if rating > 0 and self.interaction_matrix[user_idx, item_idx] == 0:
                        item_id = self.reverse_item_mapping[item_idx]
                        item_scores[item_id] = item_scores.get(item_id, 0) + similarity * rating
            
            # Get top-k recommendations
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations[user_id] = [item_id for item_id, _ in sorted_items[:k]]
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from user similarity analysis."""
        if not self.is_fitted:
            return {}
        return {'user_similarity': self.explainability_data['avg_user_similarity']}


class CollaborativeNMF(BaseRecommender):
    """
    Collaborative filtering using Non-negative Matrix Factorization.
    Medium complexity model.
    """
    
    def __init__(self, n_components: int = 100, max_iter: int = 200):
        super().__init__("collaborative_nmf", "medium", "collaborative")
        self.n_components = n_components
        self.max_iter = max_iter
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit NMF on user-item interaction matrix."""
        # Create user and item mappings
        unique_users = sorted(train_data['user_id'].unique())
        unique_items = sorted(train_data['movie_id'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Create interaction matrix (normalize to [0,1] for NMF)
        n_users = len(unique_users)
        n_items = len(unique_items)
        interaction_matrix = np.zeros((n_users, n_items))
        
        for _, row in train_data.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            item_idx = self.item_mapping[row['movie_id']]
            # Normalize rating to [0,1]
            interaction_matrix[user_idx, item_idx] = (row['rating'] - 1) / 4
        
        # Apply NMF
        self.model = NMF(n_components=self.n_components, 
                        max_iter=self.max_iter, 
                        random_state=42)
        self.user_factors = self.model.fit_transform(interaction_matrix)
        self.item_factors = self.model.components_.T
        
        self.is_fitted = True
        
        # For explainability, analyze factor importance
        user_factor_variance = np.var(self.user_factors, axis=0)
        item_factor_variance = np.var(self.item_factors, axis=0)
        
        self.explainability_data = {
            'user_factor_variance': user_factor_variance.tolist(),
            'item_factor_variance': item_factor_variance.tolist(),
            'n_components': self.n_components,
            'model_type': 'nmf',
            'approach': 'collaborative',
            'reconstruction_error': self.model.reconstruction_err_
        }
    
    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations using NMF predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        recommendations = {}
        
        for user_id in test_data['user_id'].unique():
            if user_id not in self.user_mapping:
                recommendations[user_id] = []
                continue
                
            user_idx = self.user_mapping[user_id]
            user_vector = self.user_factors[user_idx]
            
            # Calculate scores for all items
            scores = np.dot(user_vector, self.item_factors.T)
            
            # Get top-k items
            top_k_indices = np.argsort(scores)[::-1][:k]
            top_k_items = [self.reverse_item_mapping[idx] for idx in top_k_indices 
                          if idx in self.reverse_item_mapping]
            
            recommendations[user_id] = top_k_items
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from NMF factors."""
        if not self.is_fitted:
            return {}
        
        # Use factor variance as importance
        importance = {}
        for i, variance in enumerate(self.explainability_data['user_factor_variance']):
            importance[f'user_factor_{i}'] = variance
            
        return importance


class CollaborativeALS(BaseRecommender):
    """
    Collaborative filtering using Alternating Least Squares.
    High complexity model using implicit library.
    """
    
    def __init__(self, factors: int = 200, regularization: float = 0.01, iterations: int = 50):
        super().__init__("collaborative_als", "high", "collaborative")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit ALS model using implicit library."""
        try:
            import implicit
        except ImportError:
            raise ImportError("Please install implicit library: pip install implicit")
        
        # Create user and item mappings
        unique_users = sorted(train_data['user_id'].unique())
        unique_items = sorted(train_data['movie_id'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Create sparse interaction matrix
        rows, cols, data = [], [], []
        for _, row in train_data.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            item_idx = self.item_mapping[row['movie_id']]
            confidence = row['rating']  # Use rating as confidence
            
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(confidence)
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        # Fit ALS model
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42
        )
        
        # ALS expects item-user matrix (transposed)
        self.model.fit(interaction_matrix.T)
        
        self.interaction_matrix = interaction_matrix
        self.is_fitted = True
        
        # For explainability, analyze factor norms
        user_factor_norms = np.linalg.norm(self.model.user_factors, axis=1)
        item_factor_norms = np.linalg.norm(self.model.item_factors, axis=1)
        
        self.explainability_data = {
            'avg_user_factor_norm': np.mean(user_factor_norms),
            'avg_item_factor_norm': np.mean(item_factor_norms),
            'factors': self.factors,
            'model_type': 'als',
            'approach': 'collaborative'
        }
    
    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations using ALS predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        recommendations = {}
        
        for user_id in test_data['user_id'].unique():
            if user_id not in self.user_mapping:
                recommendations[user_id] = []
                continue
                
            user_idx = self.user_mapping[user_id]
            
            # Get recommendations from ALS model
            item_ids, scores = self.model.recommend(
                user_idx, 
                self.interaction_matrix[user_idx], 
                N=k
            )
            
            # Convert back to original item IDs
            recommended_items = [self.reverse_item_mapping[item_idx] 
                               for item_idx in item_ids 
                               if item_idx in self.reverse_item_mapping]
            
            recommendations[user_id] = recommended_items
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ALS factors."""
        if not self.is_fitted:
            return {}
        
        return {
            'user_factor_norm': self.explainability_data['avg_user_factor_norm'],
            'item_factor_norm': self.explainability_data['avg_item_factor_norm']
        } 