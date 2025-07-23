import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from core.data_manager import DataManager
import config

class GlobalDataManager:
    """
    Global data manager ensuring consistent splits across all experiments.
    Creates splits ONCE and reuses them for all experiments.
    """
    
    def __init__(self, force_rebuild: bool = False):
        """
        Initialize global data manager.
        
        Args:
            force_rebuild: If True, rebuild splits even if they exist
        """
        self.force_rebuild = force_rebuild
        self.splits_dir = config.SPLITS_PATH
        self.splits_dir.mkdir(exist_ok=True)
        
        # Configuration for consistent splitting
        self.config = {
            'SAMPLE_SIZE': config.SAMPLE_SIZE,
            'POSITIVE_RATING_THRESHOLD': config.POSITIVE_RATING_THRESHOLD,
            'TEST_SIZE': config.TEST_SIZE,
            'RANDOM_STATE': 42,
            'USER_GROUP_THRESHOLDS': config.USER_GROUP_THRESHOLDS,
            'DATA_PATH': config.DATA_PATH
        }
        
        self.splits_info = None
        self.train_data = None
        self.test_data = None
        self.user_groups = None
        self.data_manager = None
        
    def get_splits_signature(self) -> str:
        """Generate unique signature for current configuration."""
        signature_data = {
            'sample_size': self.config['SAMPLE_SIZE'],
            'positive_threshold': self.config['POSITIVE_RATING_THRESHOLD'],
            'test_size': self.config['TEST_SIZE'],
            'random_state': self.config['RANDOM_STATE'],
            'thresholds': self.config['USER_GROUP_THRESHOLDS']
        }
        
        # Create hash-like signature
        signature_str = json.dumps(signature_data, sort_keys=True)
        import hashlib
        return hashlib.md5(signature_str.encode()).hexdigest()[:12]
    
    def get_splits_path(self) -> Path:
        """Get path for splits based on configuration."""
        signature = self.get_splits_signature()
        sample_suffix = f"_sample_{self.config['SAMPLE_SIZE']}" if self.config['SAMPLE_SIZE'] else "_full"
        return self.splits_dir / f"splits_{signature}{sample_suffix}"
    
    def splits_exist(self) -> bool:
        """Check if splits already exist for current configuration."""
        splits_path = self.get_splits_path()
        required_files = [
            "train_data.pkl",
            "test_data.pkl", 
            "user_groups.pkl",
            "splits_info.json"
        ]
        return all((splits_path / file).exists() for file in required_files)
    
    def create_splits(self) -> None:
        """Create train/test splits and user groups."""
        print("Creating new data splits...")
        
        # Initialize data manager with current config
        self.data_manager = DataManager(self.config)
        
        # Process data
        processed_data = self.data_manager.process_data()
        
        # Create train/test split
        train_data, test_data = self.data_manager.create_train_test_split(
            processed_data, test_size=self.config['TEST_SIZE']
        )
        
        # Create user groups based on training data
        user_groups = self.data_manager.split_user_groups(train_data)
        
        # Store splits
        self.train_data = train_data
        self.test_data = test_data
        self.user_groups = user_groups
        
        # Create splits info (ensure JSON serializable)
        def make_serializable(obj):
            """Convert any object to JSON serializable format."""
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, (Path, os.PathLike)):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            else:
                # For complex objects, convert to string representation
                return str(obj)
        
        serializable_config = make_serializable(self.config)
        
        self.splits_info = {
            'created_at': datetime.now().isoformat(),
            'config': serializable_config,
            'signature': self.get_splits_signature(),
            'train_size': len(train_data),
            'test_size': len(test_data),
            'total_users': len(processed_data['user_id'].unique()),
            'total_movies': len(processed_data['movie_id'].unique()),
            'user_groups': {
                group: {
                    'count': info['count'],
                    'description': info['description']
                }
                for group, info in user_groups.items()
            }
        }
        
        print(f"Created splits: {len(train_data)} train, {len(test_data)} test")
        print(f"User groups: {[(group, info['count']) for group, info in user_groups.items()]}")
    
    def save_splits(self) -> None:
        """Save splits to disk."""
        splits_path = self.get_splits_path()
        splits_path.mkdir(exist_ok=True)
        
        print(f"Saving splits to {splits_path}")
        
        # Save data splits
        self.train_data.to_pickle(splits_path / "train_data.pkl")
        self.test_data.to_pickle(splits_path / "test_data.pkl")
        
        # Save user groups
        with open(splits_path / "user_groups.pkl", 'wb') as f:
            pickle.dump(self.user_groups, f)
        
        # Save splits info
        with open(splits_path / "splits_info.json", 'w') as f:
            json.dump(self.splits_info, f, indent=2)
        
        print(f"Splits saved successfully")
    
    def load_splits(self) -> None:
        """Load existing splits from disk."""
        splits_path = self.get_splits_path()
        
        print(f"Loading existing splits from {splits_path}")
        
        # Load data splits
        self.train_data = pd.read_pickle(splits_path / "train_data.pkl")
        self.test_data = pd.read_pickle(splits_path / "test_data.pkl")
        
        # Load user groups
        with open(splits_path / "user_groups.pkl", 'rb') as f:
            self.user_groups = pickle.load(f)
        
        # Load splits info
        with open(splits_path / "splits_info.json", 'r') as f:
            self.splits_info = json.load(f)
        
        print(f"Loaded splits: {len(self.train_data)} train, {len(self.test_data)} test")
        print(f"User groups: {[(group, info['count']) for group, info in self.user_groups.items()]}")
        print(f"Created: {self.splits_info['created_at']}")
    
    def ensure_splits_ready(self) -> None:
        """Ensure splits are ready (load existing or create new)."""
        if self.force_rebuild or not self.splits_exist():
            print("Creating new data splits...")
            self.create_splits()
            self.save_splits()
        else:
            print("Using existing data splits...")
            self.load_splits()
    
    def get_data_for_experiment(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Get consistent train/test data and user groups for any experiment.
        
        Returns:
            Tuple of (train_data, test_data, user_groups)
        """
        self.ensure_splits_ready()
        return self.train_data.copy(), self.test_data.copy(), self.user_groups.copy()
    
    def filter_data_by_user_group(self, data: pd.DataFrame, user_group: str) -> pd.DataFrame:
        """
        Filter data to specific user group using global user groups.
        
        Args:
            data: Input dataframe
            user_group: User group name
            
        Returns:
            Filtered dataframe
        """
        if user_group == "full":
            return data
        
        if user_group not in self.user_groups:
            raise ValueError(f"Unknown user group: {user_group}. Available: {list(self.user_groups.keys())}")
        
        target_users = self.user_groups[user_group]["users"]
        return data[data['user_id'].isin(target_users)]
    
    def get_splits_summary(self) -> Dict:
        """Get summary of current splits configuration."""
        if self.splits_info is None:
            self.ensure_splits_ready()
        
        return {
            'splits_info': self.splits_info,
            'consistent_across_experiments': True,
            'path': str(self.get_splits_path()),
            'signature': self.get_splits_signature()
        }
    
    def list_available_splits(self) -> List[Dict]:
        """List all available split configurations."""
        available_splits = []
        
        for splits_dir in self.splits_dir.glob("splits_*"):
            if splits_dir.is_dir() and (splits_dir / "splits_info.json").exists():
                try:
                    with open(splits_dir / "splits_info.json", 'r') as f:
                        info = json.load(f)
                    available_splits.append({
                        'path': str(splits_dir),
                        'created_at': info['created_at'],
                        'config': info['config'],
                        'signature': info['signature'],
                        'train_size': info['train_size'],
                        'test_size': info['test_size'],
                        'user_groups': info['user_groups']
                    })
                except Exception as e:
                    print(f"Failed to read {splits_dir}: {e}")
        
        return available_splits

# Global instance for easy access
global_data_manager = GlobalDataManager() 