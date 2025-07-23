import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import config as app_config
from .text_processing import TextFeatureEnricher, TRANSFORMERS_AVAILABLE

class DataManager:
    """
    Centralized data management for all recommendation approaches.
    Handles loading, preprocessing, and user group segmentation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data manager with configuration.
        
        Args:
            config: Configuration dictionary from config.py
        """
        self.config = config
        self.raw_data = None
        self.processed_data = None
        self.user_groups = {}
        self.item_features = None
        self.user_features = None
        self.interaction_matrix = None
        self.scalers = {}
        
        if TRANSFORMERS_AVAILABLE:
            self.text_enricher = TextFeatureEnricher(cache_path=app_config.DATA_PATH)
        else:
            self.text_enricher = None
        
    def load_movielens_data(self) -> pd.DataFrame:
        """
        Load the pre-processed MovieLens 1M enriched dataset.
        
        Returns:
            Processed ratings dataframe with all features
        """
        # Use the enriched dataset that's already fully processed
        enriched_path = os.path.join(self.config['DATA_PATH'], 'movielens_1m_enriched.csv')
        
        if not os.path.exists(enriched_path):
            raise FileNotFoundError(f"Enriched dataset not found at {enriched_path}")
        
        print(f"Loading enriched dataset from {enriched_path}")
        data = pd.read_csv(enriched_path)
        
        # Standardize column names (the CSV uses lowercase)
        column_mapping = {
            'userid': 'user_id', 
            'movieid': 'movie_id',
            'rating': 'rating',
            'timestamp': 'timestamp'
        }
        
        # Only rename columns that exist
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                data = data.rename(columns={old_col: new_col})
        
        # Apply positive rating threshold
        data['is_positive'] = (data['rating'] >= self.config['POSITIVE_RATING_THRESHOLD']).astype(int)
        
        # Sample data if configured
        if self.config.get('SAMPLE_SIZE') and self.config['SAMPLE_SIZE'] < len(data):
            print(f"Sampling {self.config['SAMPLE_SIZE']} records from {len(data)} total")
            data = data.sample(n=self.config['SAMPLE_SIZE'], random_state=42)
        
        print(f"Loaded dataset with {len(data)} records and {len(data.columns)} features")
        self.raw_data = data
        return data
    
    def enrich_with_text_embeddings(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches the main dataframe with pre-computed text embeddings.
        """
        if not self.text_enricher:
            print("Text enrichment skipped: Transformers library not available.")
            return data

        # Ensure embeddings are processed and saved
        movies_df = data[['movie_id', 'title', 'genres']].drop_duplicates(subset=['movie_id'])
        self.text_enricher.process_and_save_embeddings(movies_df)

        # Load embeddings
        embeddings = self.text_enricher.load_embeddings()
        if not embeddings:
            return data

        # Convert embeddings dict to a dataframe
        embedding_df = pd.DataFrame.from_dict(embeddings, orient='index')
        embedding_df.columns = [f'bert_emb_{i}' for i in range(embedding_df.shape[1])]
        embedding_df.index.name = 'movie_id'

        # Merge with the main data
        data = data.merge(embedding_df, on='movie_id', how='left')
        
        # Fill missing embeddings with zeros (for movies that might have been filtered out)
        embedding_cols = [f'bert_emb_{i}' for i in range(embedding_df.shape[1])]
        data[embedding_cols] = data[embedding_cols].fillna(0)
        
        print(f"Data enriched with {len(embedding_cols)} text embedding features.")
        return data
    
    def create_content_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create content-based features for items and users.
        
        Args:
            data: Raw ratings data
            
        Returns:
            Data with content features
        """
        # Genre features (one-hot encoding)
        genres = data['genres'].str.get_dummies(sep='|')
        genre_cols = [f'genre_{col}' for col in genres.columns]
        genres.columns = genre_cols
        
        # Movie popularity features
        movie_stats = data.groupby('movie_id').agg({
            'rating': ['mean', 'count', 'std'],
            'user_id': 'nunique'
        }).round(3)
        movie_stats.columns = ['avg_rating', 'rating_count', 'rating_std', 'unique_users']
        movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)
        movie_stats['popularity_score'] = (
            movie_stats['rating_count'] * movie_stats['avg_rating'] / 
            (movie_stats['rating_count'] + 10)  # Bayesian average
        )
        
        # User preference features
        user_stats = data.groupby('user_id').agg({
            'rating': ['mean', 'count', 'std'],
            'movie_id': 'nunique'
        }).round(3)
        user_stats.columns = ['user_avg_rating', 'user_rating_count', 'user_rating_std', 'user_unique_movies']
        user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)
        
        # Temporal features
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data['year'] = data['timestamp'].dt.year
        data['month'] = data['timestamp'].dt.month
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # Merge all features
        data = data.merge(genres, left_on='movie_id', right_index=True, how='left')
        data = data.merge(movie_stats, left_on='movie_id', right_index=True, how='left')
        data = data.merge(user_stats, left_on='user_id', right_index=True, how='left')
        
        # Encode categorical features
        categorical_cols = ['gender', 'occupation']
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
                self.scalers[f'{col}_encoder'] = le
        
        # Age binning
        if 'age' in data.columns:
            data['age_group'] = pd.cut(data['age'], bins=[0, 18, 25, 35, 45, 56], 
                                     labels=['<18', '18-24', '25-34', '35-44', '45+'])
            age_encoder = LabelEncoder()
            data['age_group_encoded'] = age_encoder.fit_transform(data['age_group'].astype(str))
            self.scalers['age_encoder'] = age_encoder
        
        return data
    
    def create_interaction_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-item interaction matrix for collaborative filtering.
        
        Args:
            data: Processed ratings data
            
        Returns:
            User-item interaction matrix
        """
        # Create explicit rating matrix
        interaction_matrix = data.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating',
            fill_value=0
        )
        
        # Create binary interaction matrix for implicit feedback
        binary_matrix = (interaction_matrix > 0).astype(int)
        
        self.interaction_matrix = {
            'explicit': interaction_matrix,
            'implicit': binary_matrix
        }
        
        return interaction_matrix
    
    def split_user_groups(self, train_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Split users into moderate, active, and power user groups based on training data.
        
        Args:
            train_data: Training data (used to determine user interaction counts)
            
        Returns:
            Dictionary with user group information
        """
        # Import thresholds from config
        from config import USER_GROUP_THRESHOLDS
        
        # Calculate user interaction counts in TRAINING data
        user_counts = train_data.groupby('user_id').size()
        
        # Get all users from the full processed data
        all_users = self.processed_data['user_id'].unique()
        
        # Define user groups based on training interaction counts using config thresholds
        moderate_users = []
        active_users = []
        power_users = []
        
        moderate_threshold = USER_GROUP_THRESHOLDS['moderate_users']
        active_threshold = USER_GROUP_THRESHOLDS['active_users']
        
        for user_id in all_users:
            count = user_counts.get(user_id, 0)  # 0 if user not in training data
            
            if count <= moderate_threshold:
                moderate_users.append(user_id)
            elif count <= active_threshold:
                active_users.append(user_id)
            else:
                power_users.append(user_id)
        
        self.user_groups = {
            'moderate_users': {
                'users': moderate_users,
                'count': len(moderate_users),
                'description': f'Moderately active users with ≤{moderate_threshold} training ratings'
            },
            'active_users': {
                'users': active_users,
                'count': len(active_users),
                'description': f'Active users with {moderate_threshold+1}-{active_threshold} training ratings'
            },
            'power_users': {
                'users': power_users,
                'count': len(power_users),
                'description': f'Power users with >{active_threshold} training ratings'
            },
            'full': {
                'users': all_users.tolist(),
                'count': len(all_users),
                'description': 'All users'
            }
        }
        
        return self.user_groups
    
    def create_train_test_split(self, data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split maintaining user groups.
        
        Args:
            data: Processed data
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        train_data = []
        test_data = []
        
        # Split by user to ensure same users in train/test
        for user_id in data['user_id'].unique():
            user_data = data[data['user_id'] == user_id]
            
            if len(user_data) == 1:
                # Single rating goes to train
                train_data.append(user_data)
            else:
                # Split user's ratings
                user_train, user_test = train_test_split(
                    user_data, test_size=test_size, 
                    random_state=42, stratify=None
                )
                train_data.append(user_train)
                test_data.append(user_test)
        
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        return train_df, test_df
    
    def get_features_for_approach(self, approach: str) -> List[str]:
        """
        Get relevant feature columns for a specific recommendation approach.
        
        Args:
            approach: 'content', 'collaborative', or 'hybrid'
            
        Returns:
            List of feature column names
        """
        if self.processed_data is None:
            raise ValueError("Data must be processed first")
        
        all_cols = self.processed_data.columns.tolist()
        
        if approach == 'content':
            # Content-based features
            content_features = [col for col in all_cols if any(prefix in col for prefix in [
                'genre_', 'avg_rating', 'rating_count', 'popularity_score',
                'user_avg_rating', 'user_rating_count', 'gender_encoded',
                'occupation_encoded', 'age_group_encoded', 'year', 'month'
            ])]
            return content_features
            
        elif approach == 'collaborative':
            # Collaborative features (user/item IDs for matrix factorization)
            return ['user_id', 'movie_id', 'rating']
            
        elif approach == 'hybrid':
            # Combination of content and collaborative
            content_features = self.get_features_for_approach('content')
            collab_features = ['user_id', 'movie_id', 'rating']
            return list(set(content_features + collab_features))
        
        else:
            raise ValueError(f"Unknown approach: {approach}")
    
    def process_data(self) -> pd.DataFrame:
        """
        Load pre-processed data and prepare for modeling.
        
        Returns:
            Processed data ready for modeling
        """
        # If already processed, return cached data
        if self.processed_data is not None:
            return self.processed_data
        
        try:
            # Process and enrich data
            data = self.load_movielens_data()
            data = self.create_content_features(data)
            
            # Add text embeddings
            data = self.enrich_with_text_embeddings(data)
            
            self.processed_data = data
            return data
            
        finally:
            print("Data processing completed:")
            if self.interaction_matrix:
                print(f"   Interaction matrix: {self.interaction_matrix['explicit'].shape}")
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save the fully processed data to a file.
        
        Args:
            output_path: Directory to save processed data
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save processed data
        self.processed_data.to_pickle(os.path.join(output_path, 'processed_data.pkl'))
        
        # Save user groups
        with open(os.path.join(output_path, 'user_groups.pkl'), 'wb') as f:
            pickle.dump(self.user_groups, f)
        
        # Save interaction matrix
        with open(os.path.join(output_path, 'interaction_matrix.pkl'), 'wb') as f:
            pickle.dump(self.interaction_matrix, f)
        
        # Save scalers/encoders
        with open(os.path.join(output_path, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f) 