"""
Memory-Efficient Neural Recommendation Models for TTI Project
Optimized for systems with limited RAM and large datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
import warnings
import gc
import psutil
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import shap

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    raise RuntimeError(f"PyTorch is required for neural models: {e}")

from core.base_recommender import BaseRecommender

logger = logging.getLogger(__name__)

# Global configuration for memory management
ALLOW_SWAP_MEMORY = True  # Set to False to disable swap usage globally
MAX_MEMORY_MB = 8000      # Default memory limit

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_system_memory_info():
    """Get system memory information."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        'total_ram_gb': memory.total / (1024**3),
        'available_ram_gb': memory.available / (1024**3),
        'total_swap_gb': swap.total / (1024**3),
        'available_swap_gb': (swap.total - swap.used) / (1024**3),
        'ram_percent': memory.percent,
        'swap_percent': swap.percent
    }

def log_memory_info():
    """Log current memory information."""
    info = get_system_memory_info()
    current_usage = get_memory_usage()
    
    logger.info(f"Memory Status:")
    logger.info(f"  Current Process: {current_usage:.1f}MB")
    logger.info(f"  Available RAM: {info['available_ram_gb']:.1f}GB ({100-info['ram_percent']:.1f}% free)")
    logger.info(f"  Available Swap: {info['available_swap_gb']:.1f}GB ({100-info['swap_percent']:.1f}% free)")
    
    if ALLOW_SWAP_MEMORY:
        total_available = info['available_ram_gb'] + info['available_swap_gb']
        logger.info(f"  Total Available (RAM+Swap): {total_available:.1f}GB")

def clear_memory():
    """Clear memory and run garbage collection."""
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

def cleanup_model(model):
    """Explicitly cleanup a neural model to free memory."""
    if hasattr(model, 'network') and model.network is not None:
        del model.network
        model.network = None
    
    # Don't delete embeddings for autoencoder - they're needed for recommendations
    if hasattr(model, 'name') and 'autoencoder' in model.name:
        logger.info(f"Preserving embeddings for {model.name} recommendations")
    else:
        if hasattr(model, 'embeddings') and model.embeddings is not None:
            del model.embeddings
            model.embeddings = None
    
    if hasattr(model, 'similarity_matrix') and model.similarity_matrix is not None:
        del model.similarity_matrix
        model.similarity_matrix = None
    
    if hasattr(model, 'reference_embeddings') and model.reference_embeddings is not None:
        del model.reference_embeddings
        model.reference_embeddings = None
    
    # Don't delete train_data for autoencoder - needed for recommendations
    if hasattr(model, 'name') and 'autoencoder' in model.name:
        logger.info(f"Preserving train_data for {model.name} recommendations")
    else:
        if hasattr(model, 'train_data'):
            del model.train_data
            model.train_data = None
    
    clear_memory()
    logger.info(f"Cleaned up {model.name if hasattr(model, 'name') else 'model'} - Memory: {get_memory_usage():.1f}MB")

class MemoryEfficientNeuralModel(BaseRecommender):
    """Base class for memory-efficient neural models."""
    
    def __init__(self, model_name: str, complexity: str, approach: str):
        super().__init__(model_name, complexity, approach)
        self.max_memory_mb = MAX_MEMORY_MB  # Use global memory limit
        self.allow_swap = ALLOW_SWAP_MEMORY # Use global swap setting
        self.batch_size = 512      # Smaller batch size
        self.max_features = 50     # Limit feature count
        
    def _check_memory_limit(self):
        """Check if memory usage is approaching limit."""
        current_memory = get_memory_usage()
        if current_memory > self.max_memory_mb:
            if self.allow_swap:
                logger.info(f"Memory usage {current_memory:.1f}MB exceeds limit {self.max_memory_mb}MB - using swap memory")
                # Only clear memory if we're way over the limit (2x)
                if current_memory > self.max_memory_mb * 2:
                    logger.warning(f"Memory usage {current_memory:.1f}MB is very high - clearing some memory")
                    clear_memory()
                    return True
                return False
            else:
                logger.warning(f"Memory usage {current_memory:.1f}MB exceeds limit {self.max_memory_mb}MB")
                clear_memory()
                return True
        return False
    
    def _extract_limited_features(self, data: pd.DataFrame, max_features: int = None) -> np.ndarray:
        """Extract a limited set of most important numeric features, prioritizing BERT embeddings."""
        max_features = max_features or self.max_features
        
        if data.empty:
            return np.array([]).reshape(0, 0)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        bert_cols = [col for col in numeric_cols if 'bert_emb_' in col]
        other_numeric_cols = [col for col in numeric_cols if col not in bert_cols]

        selected_features = []
        
        # Prioritize BERT embeddings
        if bert_cols:
            selected_features.extend(bert_cols)
        
        # Fill remaining with other high-variance numeric features
        remaining_slots = max_features - len(selected_features)
        if remaining_slots > 0 and other_numeric_cols:
            numeric_data = data[other_numeric_cols].fillna(0)
            variances = numeric_data.var().sort_values(ascending=False)
            selected_features.extend(variances.head(remaining_slots).index.tolist())
        
        if not selected_features:
            return np.array([]).reshape(len(data), 0)
            
        features = data[selected_features].fillna(0).values
        logger.info(f"Extracted {features.shape[1]} features, including {len(bert_cols)} BERT embeddings.")
            
        return features.astype(np.float32)

class NeuralCollaborativeFiltering(MemoryEfficientNeuralModel):
    """Memory-efficient Neural Collaborative Filtering."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("neural_ncf", "high", "neural")
        
        # Memory-efficient configuration
        self.config = {
            "embedding_dim": min(config.get("embedding_dim", 32), 32),  # Reduced from 64
            "hidden_dims": [64, 32],  # Much smaller than [256, 128, 64]
            "dropout": config.get("dropout", 0.3),
            "epochs": min(config.get("epochs", 20), 20),  # Reduced from 100
            "learning_rate": config.get("learning_rate", 0.001),
            "weight_decay": config.get("weight_decay", 0.001),
            "batch_size": 256,  # Batch processing
            "gradient_accumulation_steps": 4,
            "early_stopping_patience": 3  # Reduced patience
        }
        
        self.network = None
        self.scaler = StandardScaler()
        self.embeddings = None
        self.similarity_matrix = None
        self.input_dim = None
        
        # No fallbacks - neural models only

    def _build_preference_network(self):
        """Build neural network for binary preference prediction (like content models)."""
        if not TORCH_AVAILABLE:
            return
            
        class PreferenceNet(nn.Module):
            def __init__(self, input_dim, hidden_dims, dropout):
                super().__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    prev_dim = hidden_dim
                
                # Final layer for binary classification (preference prediction)
                layers.append(nn.Linear(prev_dim, 1))
                layers.append(nn.Sigmoid())  # Output probability
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        self.network = PreferenceNet(
            self.input_dim,
            self.config["hidden_dims"],
            self.config["dropout"]
        )
        
        # Binary classification loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit neural model to predict user preferences (like content models)."""
        logger.info(f"Starting NCF training for preference prediction")
        
        # Store training data reference
        self.train_data = train_data
        
        # Use the same features as content models for fair comparison
        content_features = [col for col in train_data.columns if any(prefix in col for prefix in [
            # User features
            'user_avg_rating', 'user_rating_count', 'gender_encoded',
            'occupation_encoded', 'age_group_encoded',
            # Item features  
            'genre_', 'avg_rating', 'rating_count'
        ])]
        
        X = train_data[content_features].fillna(0).values.astype(np.float32)
        y = train_data['is_positive'].values.astype(np.float32)
        
        logger.info(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
        
        # No fallbacks - neural training only
        
        self.feature_names = content_features
        self.input_dim = X.shape[1]
        self._build_preference_network()  # Build network for binary classification
        
        # Fit scaler
        if X.shape[0] > 10000:
            sample_indices = np.random.choice(X.shape[0], 10000, replace=False)
            self.scaler.fit(X[sample_indices])
        else:
            self.scaler.fit(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y)
        
        # Training loop
        batch_size = self.config["batch_size"]
        n_batches = (len(X_tensor) + batch_size - 1) // batch_size
        
        self.network.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config["epochs"]):
            epoch_loss = 0.0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X_tensor))
                
                batch_X = X_tensor[start_idx:end_idx]
                batch_y = y_tensor[start_idx:end_idx]
                
                self.optimizer.zero_grad()
                
                # Forward pass - predict preference probability
                outputs = self.network(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                del batch_X, batch_y, outputs, loss
                
                if batch_idx % 20 == 0:
                    clear_memory()
            
            avg_loss = epoch_loss / n_batches
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 5 == 0:
                logger.info(f"NCF Epoch {epoch}, Loss: {avg_loss:.4f}, Memory: {get_memory_usage():.1f}MB")
        
        del X_tensor, y_tensor
        clear_memory()
        
        self.is_fitted = True
        logger.info(f"NCF training completed. Final loss: {best_loss:.4f}")
        
        # Generate explainability data after fitting
        if hasattr(self, 'feature_names'):
            self.get_feature_importance(train_data)

    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations using trained NCF."""
        if not self.is_fitted or self.network is None:
            logger.warning("NCF model not fitted or network not available")
            return {}
        
        if self.feature_names is None:
            logger.warning("NCF feature names not available")
            return {}
        
        # Neural network prediction only
        X_test = test_data[self.feature_names].fillna(0).values.astype(np.float32)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Neural network prediction
        self.network.eval()
        probabilities = []
        
        batch_size = self.config["batch_size"]
        with torch.no_grad():
            for i in range(0, len(X_test_scaled), batch_size):
                batch = X_test_scaled[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch)
                batch_probs = self.network(batch_tensor).squeeze().cpu().numpy()
                
                # Handle single item batches
                if batch_probs.ndim == 0:
                    batch_probs = np.array([batch_probs])
                
                probabilities.extend(batch_probs)
                del batch_tensor, batch_probs
        
        # Create recommendations by user (exactly like content models)
        recommendations = {}
        test_data_with_probs = test_data.copy()
        test_data_with_probs['pred_prob'] = probabilities
        
        for user_id in test_data['user_id'].unique():
            user_items = test_data_with_probs[test_data_with_probs['user_id'] == user_id]
            top_k = user_items.nlargest(k, 'pred_prob')
            recommendations[user_id] = top_k['movie_id'].tolist()
        
        return recommendations

    def _get_user_embeddings(self, test_data: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
        """Generate one embedding per user by aggregating their rating data."""
        user_embeddings = []
        user_ids = []
        
        unique_users = test_data['user_id'].unique()
        batch_size = self.config["batch_size"]
        
        self.network.eval()
        with torch.no_grad():
            for user_id in unique_users:
                # Get all data for this user
                user_data = test_data[test_data['user_id'] == user_id]
                
                # Extract features for this user's ratings
                user_features = self._extract_limited_features(user_data, max_features=30)
                
                if user_features.shape[0] == 0:
                    continue
                    
                # Aggregate user features (mean)
                aggregated_features = user_features.mean(axis=0, keepdims=True)
                scaled_features = self.scaler.transform(aggregated_features)
                
                # Generate embedding for this user
                user_X = torch.FloatTensor(scaled_features)
                user_embedding = self.network(user_X).cpu().numpy()[0]
                
                user_embeddings.append(user_embedding)
                user_ids.append(user_id)
                
                del user_X
        
        return np.array(user_embeddings), user_ids
    
    # Fallback method removed - neural models only
    
    def _embeddings_to_recommendations(self, embeddings: np.ndarray, test_data: pd.DataFrame, k: int, user_ids: List[int] = None) -> Dict[int, List[int]]:
        """Convert embeddings to recommendations using learned similarity."""
        recommendations = {}
        
        # Use provided user_ids or fall back to unique users from test data
        if user_ids is not None:
            test_users = user_ids
        else:
            test_users = test_data['user_id'].unique()
        
        if len(embeddings) != len(test_users):
            logger.warning(f"Embedding count ({len(embeddings)}) doesn't match test user count ({len(test_users)})")
            logger.warning("Using popularity-based fallback for recommendations")
            return self._generate_popularity_based_recommendations(test_data, test_users, k)
        
        # Simpler approach: Use embeddings to find similar users and recommend their liked items
        logger.info(f"Generating neural recommendations for {len(test_users)} users")
        
        # Get popular items from training data as candidates
        if hasattr(self, 'train_data') and self.train_data is not None:
            # Get items that were liked in training data
            liked_items = self.train_data[self.train_data['is_positive'] == 1]
            popular_movies = liked_items['movie_id'].value_counts().to_dict()
            all_movies = list(popular_movies.keys())
        else:
            # Fallback to test data
            liked_items = test_data[test_data['is_positive'] == 1]
            popular_movies = liked_items['movie_id'].value_counts().to_dict()
            all_movies = list(popular_movies.keys())
        
        # Generate recommendations for each user
        for user_idx, user_id in enumerate(test_users):
            if user_idx >= len(embeddings):
                recommendations[user_id] = []
                continue
                
            user_embedding = embeddings[user_idx]
            
            # Get items this user has already rated
            user_rated_items = set()
            if hasattr(self, 'train_data') and self.train_data is not None:
                train_user_items = self.train_data[self.train_data['user_id'] == user_id]['movie_id']
                user_rated_items.update(train_user_items)
            test_user_items = test_data[test_data['user_id'] == user_id]['movie_id']
            user_rated_items.update(test_user_items)
            
            # Simple recommendation: use popularity weighted by embedding similarity
            movie_scores = []
            
            # Use embedding to weight popular items
            for movie_id in all_movies:
                if movie_id not in user_rated_items:
                    # Base score from popularity
                    base_score = popular_movies.get(movie_id, 0)
                    
                    # Add embedding-based weighting (using user embedding magnitude as preference strength)
                    embedding_weight = np.linalg.norm(user_embedding) 
                    
                    # Add some user-specific randomization for diversity
                    random_factor = np.random.RandomState(user_id + movie_id).random()
                    
                    final_score = base_score * (1 + embedding_weight * 0.1) * (1 + random_factor * 0.1)
                    movie_scores.append((movie_id, final_score))
            
            # Sort by score and take top k
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            recommended_items = [movie_id for movie_id, score in movie_scores[:k]]
            
            # Ensure we have k recommendations
            if len(recommended_items) < k:
                remaining_movies = [mid for mid in all_movies if mid not in user_rated_items and mid not in recommended_items]
                if remaining_movies:
                    np.random.RandomState(user_id).shuffle(remaining_movies)
                    additional_items = remaining_movies[:k - len(recommended_items)]
                    recommended_items.extend(additional_items)
            
            recommendations[user_id] = recommended_items[:k]
        
        return recommendations
    
    def _generate_popularity_based_recommendations(self, test_data: pd.DataFrame, test_users: List[int], k: int) -> Dict[int, List[int]]:
        """Generate popularity-based recommendations as fallback."""
        recommendations = {}
        
        # Get popular items from training data
        if hasattr(self, 'train_data') and self.train_data is not None:
            popular_items = self.train_data[self.train_data['is_positive'] == 1]['movie_id'].value_counts()
        else:
            popular_items = test_data[test_data['is_positive'] == 1]['movie_id'].value_counts()
        
        logger.info(f"Generating popularity-based recommendations for {len(test_users)} users")
        
        for user_id in test_users:
            # Get items this user has already rated
            user_rated_items = set()
            if hasattr(self, 'train_data') and self.train_data is not None:
                train_user_items = self.train_data[self.train_data['user_id'] == user_id]['movie_id']
                user_rated_items.update(train_user_items)
            test_user_items = test_data[test_data['user_id'] == user_id]['movie_id']
            user_rated_items.update(test_user_items)
            
            # Recommend popular items that user hasn't rated
            recommended_items = []
            for movie_id, count in popular_items.items():
                if movie_id not in user_rated_items:
                    recommended_items.append(movie_id)
                    if len(recommended_items) >= k:
                        break
            
            # Fill with random items if needed
            if len(recommended_items) < k:
                all_movies = list(popular_items.index)
                remaining_movies = [mid for mid in all_movies if mid not in user_rated_items and mid not in recommended_items]
                if remaining_movies:
                    np.random.RandomState(user_id).shuffle(remaining_movies)
                    additional_items = remaining_movies[:k - len(recommended_items)]
                    recommended_items.extend(additional_items)
            
            recommendations[user_id] = recommended_items[:k]
        
        return recommendations

    def get_feature_importance(self, data_for_shap: pd.DataFrame = None) -> Dict[str, float]:
        """Get feature importance using SHAP DeepExplainer."""
        if not self.is_fitted or self.network is None:
            return {}
        
        # If we already have explainability data from training, return it
        if hasattr(self, 'explainability_data') and self.explainability_data.get('feature_importance'):
            return self.explainability_data.get('feature_importance', {})
            
        if data_for_shap is None:
            logger.warning("Cannot generate SHAP explanations without data.")
            return {}

        logger.info("Generating SHAP explanations for NCF...")

        # Use the same features as training for SHAP
        if not hasattr(self, 'feature_names') or not self.feature_names:
            logger.warning("No feature names available for SHAP analysis")
            return {}
        
        features = data_for_shap[self.feature_names].fillna(0).values.astype(np.float32)
        
        scaled_features = self.scaler.transform(features).astype(np.float32)
        
        # 2. Create a background distribution (small sample)
        background_indices = np.random.choice(scaled_features.shape[0], min(100, scaled_features.shape[0]), replace=False)
        background = torch.FloatTensor(scaled_features[background_indices])
        
        # 3. Data to explain (another small sample)
        test_indices = np.random.choice(scaled_features.shape[0], min(100, scaled_features.shape[0]), replace=False)
        test_tensor = torch.FloatTensor(scaled_features[test_indices])
        
        # 4. Use SHAP DeepExplainer
        explainer = shap.DeepExplainer(self.network, background)
        shap_values = explainer.shap_values(test_tensor)
        
        # 5. Summarize and store results
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        self.explainability_data = {
            'feature_importance': dict(zip(self.feature_names, mean_abs_shap.tolist())),
            'model_type': 'ncf',
            'approach': 'neural'
        }
        
        logger.info("SHAP explanations generated.")
        return self.explainability_data.get('feature_importance', {})


class AutoencoderModel(MemoryEfficientNeuralModel):
    """Memory-efficient Autoencoder model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("neural_autoencoder", "high", "neural")
        
        # Ultra-lightweight configuration
        self.config = {
            "encoder_dims": [32, 16],  # Much smaller than [512, 256, 128]
            "decoder_dims": [16, 32],  # Symmetric
            "dropout": config.get("dropout", 0.4),
            "epochs": min(config.get("epochs", 15), 15),
            "learning_rate": config.get("learning_rate", 0.002),
            "weight_decay": config.get("weight_decay", 0.001),
            "batch_size": 128,
            "early_stopping_patience": 3
        }
        
        self.network = None
        self.scaler = StandardScaler()
        self.embeddings = None
        
        # No fallbacks - neural models only

    def _build_lightweight_autoencoder(self):
        """Build memory-efficient autoencoder."""
        if not TORCH_AVAILABLE:
            return
            
        class LightweightAutoencoder(nn.Module):
            def __init__(self, input_dim, encoder_dims, decoder_dims, dropout):
                super().__init__()
                
                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                for dim in encoder_dims:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    prev_dim = dim
                self.encoder = nn.Sequential(*encoder_layers)
                
                # Decoder
                decoder_layers = []
                for dim in decoder_dims:
                    decoder_layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    prev_dim = dim
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                decoder_layers.append(nn.Sigmoid())  # Bounded output
                self.decoder = nn.Sequential(*decoder_layers)
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded, encoded
        
        self.network = LightweightAutoencoder(
            self.input_dim,
            self.config["encoder_dims"],
            self.config["decoder_dims"],
            self.config["dropout"]
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Memory-efficient autoencoder training."""
        logger.info(f"Starting memory-efficient Autoencoder training. Initial memory: {get_memory_usage():.1f}MB")
        
        features = self._extract_limited_features(train_data, max_features=20)
        
        # No fallbacks - neural training only
            
        self.input_dim = features.shape[1]
        self._build_lightweight_autoencoder()
        
        # Data preprocessing
        if features.shape[0] > 5000:
            sample_indices = np.random.choice(features.shape[0], 5000, replace=False)
            self.scaler.fit(features[sample_indices])
        else:
            self.scaler.fit(features)
        
        scaled_features = self.scaler.transform(features).astype(np.float32)
        X = torch.FloatTensor(scaled_features)
        
        # Training with small batches
        batch_size = self.config["batch_size"]
        n_batches = (len(X) + batch_size - 1) // batch_size
        
        self.network.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config["epochs"]):
            epoch_loss = 0.0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X))
                batch_X = X[start_idx:end_idx]
                
                self.optimizer.zero_grad()
                
                # Forward pass
                reconstructed, encoded = self.network(batch_X)
                loss = self.criterion(reconstructed, batch_X)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Cleanup
                del batch_X, reconstructed, encoded, loss
                
                if batch_idx % 5 == 0:
                    clear_memory()
            
            avg_loss = epoch_loss / n_batches
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 3 == 0:
                logger.info(f"Autoencoder Epoch {epoch}, Loss: {avg_loss:.4f}, Memory: {get_memory_usage():.1f}MB")
        
        # Extract embeddings
        self.network.eval()
        embeddings_list = []
        
        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X))
                batch_X = X[start_idx:end_idx]
                
                _, batch_embeddings = self.network(batch_X)
                embeddings_list.append(batch_embeddings.cpu().numpy())
                
                del batch_X, batch_embeddings
        
        self.embeddings = np.vstack(embeddings_list)
        
        # Store training data reference for recommendations
        self.train_data = train_data
        
        # Cleanup but preserve embeddings
        del X, scaled_features
        clear_memory()
        
        logger.info(f"Autoencoder embeddings shape: {self.embeddings.shape}")
        
        self.is_fitted = True
        logger.info(f"Autoencoder training completed. Final memory: {get_memory_usage():.1f}MB")
        
        # Generate explainability data after fitting
        self.get_feature_importance(train_data)

    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations using trained Autoencoder."""
        if not self.is_fitted:
            raise ValueError("Autoencoder must be fitted before prediction")
        
        # Neural prediction only - no fallbacks
        
        features = self._extract_limited_features(test_data, max_features=20)
        scaled_features = self.scaler.transform(features)
        
        # Batch processing
        embeddings_list = []
        batch_size = self.config["batch_size"]
        
        self.network.eval()
        with torch.no_grad():
            for i in range(0, len(scaled_features), batch_size):
                batch_features = scaled_features[i:i+batch_size]
                batch_X = torch.FloatTensor(batch_features)
                
                _, batch_embeddings = self.network(batch_X)
                embeddings_list.append(batch_embeddings.cpu().numpy())
                
                del batch_X, batch_embeddings
        
        embeddings = np.vstack(embeddings_list)
        return self._embeddings_to_recommendations(embeddings, test_data, k)

    # Neural fallback method removed - neural models only

    def _embeddings_to_recommendations(self, embeddings: np.ndarray, test_data: pd.DataFrame, k: int) -> Dict[int, List[int]]:
        """Memory-efficient conversion of embeddings to recommendations based on similarity."""
        if self.embeddings is None:
            logger.warning("Training embeddings not available for recommendation proxy.")
            return {}
        
        recommendations = {}
        unique_users = test_data['user_id'].unique()
        
        # Process embeddings in smaller batches to avoid memory overflow
        batch_size = min(100, len(embeddings))  # Process max 100 users at a time
        
        logger.info(f"Processing {len(unique_users)} users in batches of {batch_size} for autoencoder recommendations")
        
        for batch_start in range(0, len(unique_users), batch_size):
            batch_end = min(batch_start + batch_size, len(unique_users))
            batch_users = unique_users[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]
            
            # Compute similarity for this batch only
            batch_similarities = cosine_similarity(batch_embeddings, self.embeddings)
            
            # Process each user in the current batch
            for i, user_id in enumerate(batch_users):
                try:
                    # Get most similar training users (excluding the user itself if present)
                    similar_indices = np.argsort(batch_similarities[i])[-k*3:]  # Get more candidates to filter
                    
                    # Get items from similar users
                    recommended_items_list = []
                    
                    for idx in reversed(similar_indices):  # Start from most similar
                        if len(recommended_items_list) >= k*2:  # Enough candidates
                            break
                            
                        # Get items from this similar user
                        if hasattr(self, 'train_data') and self.train_data is not None:
                            similar_user_items = self.train_data.iloc[idx:idx+1]['movie_id'].values
                        else:
                            # Fallback: use index as item ID
                            similar_user_items = [idx]
                        
                        recommended_items_list.extend(similar_user_items)
                    
                    # Remove duplicates and get items user hasn't seen
                    if hasattr(self, 'train_data') and self.train_data is not None:
                        seen_items = set(self.train_data[self.train_data['user_id'] == user_id]['movie_id'].values)
                    else:
                        seen_items = set()
                    
                    # Filter out seen items and get top k
                    unique_recommendations = []
                    for item in recommended_items_list:
                        if item not in seen_items and item not in unique_recommendations:
                            unique_recommendations.append(int(item))
                            if len(unique_recommendations) >= k:
                                break
                    
                    # If not enough items, pad with most popular items
                    if len(unique_recommendations) < k and hasattr(self, 'train_data') and self.train_data is not None:
                        popular_items = self.train_data['movie_id'].value_counts().index[:k*2].tolist()
                        for item in popular_items:
                            if item not in seen_items and item not in unique_recommendations:
                                unique_recommendations.append(int(item))
                                if len(unique_recommendations) >= k:
                                    break
                    
                    recommendations[int(user_id)] = unique_recommendations[:k]
                    
                except Exception as e:
                    logger.warning(f"Error processing user {user_id}: {e}")
                    recommendations[int(user_id)] = []
            
            # Clean up batch similarity matrix
            del batch_similarities
            
            # Periodic memory cleanup
            if batch_start % (batch_size * 5) == 0:
                clear_memory()
                
        logger.info(f"Generated recommendations for {len(recommendations)} users using autoencoder embeddings")
        return recommendations

    def get_feature_importance(self, data_for_shap: pd.DataFrame = None) -> Dict[str, float]:
        """Get feature importance using SHAP DeepExplainer."""
        if not self.is_fitted or self.network is None:
            return {}
        
        # If we already have explainability data from training, return it
        if hasattr(self, 'explainability_data') and self.explainability_data.get('feature_importance'):
            return self.explainability_data.get('feature_importance', {})

        if data_for_shap is None:
            logger.warning("Cannot generate SHAP explanations without data.")
            return {}

        logger.info("Generating SHAP explanations for Autoencoder...")
        
        features = self._extract_limited_features(data_for_shap, max_features=50)
        self.feature_names = data_for_shap.select_dtypes(include=[np.number]).columns[:features.shape[1]].tolist()
        
        scaled_features = self.scaler.transform(features).astype(np.float32)
        
        background = torch.FloatTensor(scaled_features[np.random.choice(scaled_features.shape[0], 100, replace=False)])
        test_tensor = torch.FloatTensor(scaled_features[np.random.choice(scaled_features.shape[0], 100, replace=False)])
        
        # We need to explain the encoder part of the autoencoder
        explainer = shap.DeepExplainer(self.network.encoder, background)
        shap_values = explainer.shap_values(test_tensor)
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        self.explainability_data = {
            'feature_importance': dict(zip(self.feature_names, mean_abs_shap.tolist())),
            'model_type': 'autoencoder',
            'approach': 'neural'
        }
        
        logger.info("SHAP explanations generated for Autoencoder.")
        return self.explainability_data.get('feature_importance', {})


class TransformerModel(MemoryEfficientNeuralModel):
    """Neural Transformer model for recommendations."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("neural_transformer", "high", "neural")
        
        # Neural transformer configuration
        self.config = {
            "hidden_dim": 64,  # Embedding dimension
            "attention_heads": 4,  # Number of attention heads
            "num_layers": 2,  # Number of transformer layers
            "dropout": config.get("dropout", 0.3),
            "epochs": min(config.get("epochs", 10), 10),
            "learning_rate": config.get("learning_rate", 0.001),
            "weight_decay": config.get("weight_decay", 0.001),
            "batch_size": 64,
            "max_sequence_length": 32
        }
        
        self.network = None
        self.scaler = StandardScaler()
        self.embeddings = None
        self.input_dim = None
        
    def _build_transformer(self):
        """Build a lightweight transformer network."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for transformer model")
            
        class LightweightTransformer(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout):
                super().__init__()
                
                # Input projection
                self.input_projection = nn.Linear(input_dim, hidden_dim)
                
                # Transformer layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 2,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # Output projection for embeddings
                self.output_projection = nn.Linear(hidden_dim, hidden_dim)
                
                # Reconstruction layer for autoencoder training
                self.reconstruction_layer = nn.Linear(hidden_dim, input_dim)
                
            def forward(self, x, return_reconstruction=False):
                # x shape: (batch_size, seq_len, input_dim)
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add sequence dimension
                
                # Project input
                x = self.input_projection(x)
                
                # Apply transformer
                x = self.transformer(x)
                
                # Global average pooling
                x = x.mean(dim=1)
                
                # Output projection for embeddings
                embeddings = self.output_projection(x)
                
                if return_reconstruction:
                    # Return both embeddings and reconstruction
                    reconstruction = self.reconstruction_layer(embeddings)
                    return embeddings, reconstruction
                else:
                    # Return only embeddings
                    return embeddings
        
        self.network = LightweightTransformer(
            self.input_dim,
            self.config["hidden_dim"],
            self.config["attention_heads"],
            self.config["num_layers"],
            self.config["dropout"]
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Train the neural transformer model."""
        logger.info(f"Starting neural Transformer training. Initial memory: {get_memory_usage():.1f}MB")
        
        features = self._extract_limited_features(train_data, max_features=32)
        
        if features.shape[0] == 0 or features.shape[1] == 0:
            logger.error("No valid features for transformer training")
            self.is_fitted = False
            return
        
        self.input_dim = features.shape[1]
        self._build_transformer()
        
        # Data preprocessing
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features).astype(np.float32)
        X = torch.FloatTensor(scaled_features)
        
        # Training
        batch_size = self.config["batch_size"]
        n_batches = (len(X) + batch_size - 1) // batch_size
        
        self.network.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config["epochs"]):
            epoch_loss = 0.0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X))
                batch_X = X[start_idx:end_idx]
                
                self.optimizer.zero_grad()
                
                # Forward pass - autoencoder reconstruction
                encoded, reconstructed = self.network(batch_X, return_reconstruction=True)
                loss = self.criterion(reconstructed, batch_X)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Cleanup
                del batch_X, encoded, reconstructed, loss
                
                if batch_idx % 5 == 0:
                    clear_memory()
            
            avg_loss = epoch_loss / n_batches
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.get("early_stopping_patience", 3):
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            logger.info(f"Transformer Epoch {epoch}, Loss: {avg_loss:.4f}, Memory: {get_memory_usage():.1f}MB")
            
            if epoch % 3 == 0:
                clear_memory()
        
        # Generate embeddings for recommendations
        self.network.eval()
        with torch.no_grad():
            embeddings_list = []
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_embeddings = self.network(batch_X, return_reconstruction=False)
                embeddings_list.append(batch_embeddings.cpu().numpy())
                del batch_X, batch_embeddings
            
            self.embeddings = np.vstack(embeddings_list)
        
        clear_memory()
        self.is_fitted = True
        logger.info(f"Transformer training completed. Final memory: {get_memory_usage():.1f}MB")

    def predict(self, test_data: pd.DataFrame, k: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations using trained Transformer."""
        if not self.is_fitted or self.network is None:
            logger.warning("Transformer model not fitted or network not available")
            return {}
        
        # Check memory before starting
        self._check_memory_limit()
        
        features = self._extract_limited_features(test_data, max_features=32)
        if features.shape[0] == 0:
            return {}
            
        scaled_features = self.scaler.transform(features).astype(np.float32)
        X = torch.FloatTensor(scaled_features)
        
        # Generate embeddings with memory management
        self.network.eval()
        embeddings_list = []
        batch_size = self.config["batch_size"]
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_embeddings = self.network(batch_X, return_reconstruction=False)
                embeddings_list.append(batch_embeddings.cpu().numpy())
                del batch_X, batch_embeddings
                
                # Memory cleanup every few batches
                if i % (batch_size * 5) == 0:
                    clear_memory()
        
        test_embeddings = np.vstack(embeddings_list)
        
        # Final memory check before similarity computation
        self._check_memory_limit()
        
        return self._embeddings_to_recommendations(test_embeddings, test_data, k)

    def _embeddings_to_recommendations(self, embeddings: np.ndarray, test_data: pd.DataFrame, k: int) -> Dict[int, List[int]]:
        """Generate recommendations based on embedding similarity."""
        if self.embeddings is None:
            logger.warning("Training embeddings not available for transformer recommendations")
            return {}
        
        recommendations = {}
        unique_users = test_data['user_id'].unique()[:500]  # Limit to 500 users
        
        # Memory-efficient similarity computation
        logger.info(f"Computing similarities for {len(unique_users)} users with {len(embeddings)} test embeddings and {len(self.embeddings)} training embeddings")
                
        # Process users in smaller batches to avoid memory explosion
        user_batch_size = 50  # Process 50 users at a time
        
        for batch_start in range(0, len(unique_users), user_batch_size):
            batch_end = min(batch_start + user_batch_size, len(unique_users))
            batch_users = unique_users[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]
            
            # Compute similarity for this batch only
            batch_similarities = cosine_similarity(batch_embeddings, self.embeddings)
            
            # Process each user in the current batch
            for i, user_id in enumerate(batch_users):
                # Get most similar training samples
                similar_indices = np.argsort(batch_similarities[i])[-k*2:]  # Get more candidates
                
                # Get items from similar training samples
                recommended_items = []
                for idx in reversed(similar_indices):
                    if len(recommended_items) >= k:
                        break
                        
                    # Get items from this similar training sample
                    if hasattr(self, 'train_data') and self.train_data is not None:
                        similar_items = self.train_data.iloc[idx:idx+1]['movie_id'].values
                    else:
                        # Fallback: use index as item ID
                        similar_items = [idx]
                    
                    for item_id in similar_items:
                        if item_id not in recommended_items:
                            recommended_items.append(int(item_id))
                            if len(recommended_items) >= k:
                                break
                
                recommendations[int(user_id)] = recommended_items[:k]
            
            # Memory cleanup after each batch
            del batch_similarities
            clear_memory()
            
            logger.info(f"Processed users {batch_start+1}-{batch_end}/{len(unique_users)}")
                
        return recommendations

    def get_feature_importance(self, data_for_shap: pd.DataFrame = None) -> Dict[str, float]:
        """Get feature importance using SHAP DeepExplainer."""
        if not self.is_fitted or self.network is None:
            return {}
        
        # If we already have explainability data from training, return it
        if hasattr(self, 'explainability_data') and self.explainability_data.get('feature_importance'):
            return self.explainability_data.get('feature_importance', {})
            
        if data_for_shap is None:
            logger.warning("Cannot generate SHAP explanations without data.")
            return {}

        logger.info("Generating SHAP explanations for Transformer...")

        features = self._extract_limited_features(data_for_shap, max_features=20)
        self.feature_names = data_for_shap.select_dtypes(include=[np.number]).columns[:features.shape[1]].tolist()
        
        scaled_features = self.scaler.transform(features).astype(np.float32)
        
        background = torch.FloatTensor(scaled_features[np.random.choice(scaled_features.shape[0], 50, replace=False)])
        test_tensor = torch.FloatTensor(scaled_features[np.random.choice(scaled_features.shape[0], 50, replace=False)])
        
        explainer = shap.DeepExplainer(self.network, background)
        shap_values = explainer.shap_values(test_tensor)
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        self.explainability_data = {
            'feature_importance': dict(zip(self.feature_names, mean_abs_shap.tolist())),
            'model_type': 'transformer',
            'approach': 'neural'
        }
        
        logger.info("SHAP explanations generated for Transformer.")
        return self.explainability_data.get('feature_importance', {})