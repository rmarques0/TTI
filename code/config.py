"""
Configuration file for TTI recommendation experiments
"""
from typing import List
from pathlib import Path

# =============================================================================
# DATA PATHS AND BASIC SETTINGS
# =============================================================================

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "datasets" 
RESULTS_PATH = PROJECT_ROOT / "results"
MODELS_PATH = PROJECT_ROOT / "models"
SPLITS_PATH = PROJECT_ROOT / "data_splits"

# Data files
DATASET_FILE = DATA_PATH / "movielens_1m_enriched.csv"

# Basic experimental settings
RANDOM_STATE = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Rating thresholds
POSITIVE_RATING_THRESHOLD = 3.5

# Evaluation settings  
K_VALUES = [5, 10, 20]
DEFAULT_K = 10

# Data sampling
SAMPLE_SIZE = None  # None for full dataset, integer for sample size

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# Model Registry - Maps model names to their classes and configurations
MODEL_REGISTRY = {
    # CONTENT-BASED MODELS
    "content_logistic": {
        "class_path": "approaches.content_based.ContentBasedLogistic",
        "complexity": "low",
        "approach": "content",
        "params": {}
    },
    "content_knn": {
        "class_path": "approaches.content_based.ContentBasedKNN", 
        "complexity": "low",
        "approach": "content",
        "params": {"n_neighbors": 50}
    },
    "content_similarity": {
        "class_path": "approaches.content_based.ContentBasedSimilarity",
        "complexity": "low", 
        "approach": "content",
        "params": {}
    },
    "content_rf": {
        "class_path": "approaches.content_based.ContentBasedRandomForest",
        "complexity": "medium",
        "approach": "content", 
        "params": {"n_estimators": 100}
    },
    "content_gb": {
        "class_path": "approaches.content_based.ContentBasedGradientBoosting",
        "complexity": "medium",
        "approach": "content",
        "params": {"n_estimators": 100}
    },
    
    # COLLABORATIVE FILTERING MODELS
    "collaborative_svd": {
        "class_path": "approaches.collaborative_filtering.CollaborativeSVD",
        "complexity": "low",
        "approach": "collaborative",
        "params": {"n_components": 50}
    },
    "collaborative_user_knn": {
        "class_path": "approaches.collaborative_filtering.CollaborativeUserBased", 
        "complexity": "low",
        "approach": "collaborative",
        "params": {"n_neighbors": 50}
    },
    "collaborative_nmf": {
        "class_path": "approaches.collaborative_filtering.CollaborativeNMF",
        "complexity": "medium",
        "approach": "collaborative",
        "params": {"n_components": 100, "max_iter": 200}
    },
    
    # NEURAL/HIGH COMPLEXITY MODELS
    "neural_ncf": {
        "class_path": "approaches.neural.NeuralCollaborativeFiltering",
        "complexity": "high",
        "approach": "neural",
        "params": {
            "embedding_dim": 8,
            "hidden_dims": [32, 16],
            "dropout": 0.3,
            "epochs": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.0001
        }
    },
    "neural_autoencoder": {
        "class_path": "approaches.neural.AutoencoderModel", 
        "complexity": "high",
        "approach": "neural",
        "params": {
            "encoder_dims": [64, 32],
            "decoder_dims": [32, 64], 
            "dropout": 0.4,
            "epochs": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.0001
        }
    },
    "neural_transformer": {
        "class_path": "approaches.neural.TransformerModel",
        "complexity": "high",
        "approach": "neural",
        "params": {
            "attention_heads": 2,
            "transformer_layers": 1,
            "hidden_dim": 32,
            "dropout": 0.2,
            "epochs": 10,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001
        }
    },
    
    # HYBRID MODELS (to be implemented)
    "hybrid_weighted": {
        "class_path": "approaches.hybrid.HybridWeighted",
        "complexity": "medium", 
        "approach": "hybrid",
        "params": {"content_weight": 0.5, "collaborative_weight": 0.5}
    },
    "hybrid_stacking": {
        "class_path": "approaches.hybrid.HybridStacking",
        "complexity": "high",
        "approach": "hybrid", 
        "params": {"meta_learner": "logistic"}
    }
}

# Experiments to run - easily configurable
EXPERIMENT_CONFIGS = {
    "cross_approach_comparison": {
        "description": "Compare different recommendation approaches",
        "models": ["content_logistic", "content_rf", "collaborative_svd", "collaborative_nmf"],
        "user_groups": ["moderate_users", "active_users", "power_users", "full"],
        "enabled": True
    },
    
    "content_complexity_analysis": {
        "description": "Analyze complexity within content-based approaches", 
        "models": ["content_logistic", "content_knn", "content_rf", "content_gb"],
        "user_groups": ["moderate_users", "active_users", "power_users", "full"],
        "enabled": True
    },
    
    "collaborative_complexity_analysis": {
        "description": "Analyze complexity within collaborative approaches",
        "models": ["collaborative_svd", "collaborative_user_knn", "collaborative_nmf"],
        "user_groups": ["moderate_users", "active_users", "power_users", "full"],
        "enabled": True
    },
    
    "neural_complexity_analysis": {
        "description": "Analyze high-complexity neural approaches",
        "models": ["neural_ncf", "neural_autoencoder", "neural_transformer"],
        "user_groups": ["moderate_users", "active_users", "power_users", "full"],
        "enabled": True
    },
    
    "activity_level_analysis": {
        "description": "Analyze how user activity level affects recommendation quality",
        "models": ["content_logistic", "content_rf", "collaborative_svd", "hybrid_weighted"],
        "user_groups": ["moderate_users", "active_users", "power_users"],
        "enabled": True
    }
}

# Quick model selection for different experiment types
COMPLEXITY_LEVELS = {
    "low": [model for model, config in MODEL_REGISTRY.items() if config["complexity"] == "low"],
    "medium": [model for model, config in MODEL_REGISTRY.items() if config["complexity"] == "medium"], 
    "high": [model for model, config in MODEL_REGISTRY.items() if config["complexity"] == "high"]
}

APPROACH_TYPES = {
    "content": [model for model, config in MODEL_REGISTRY.items() if config["approach"] == "content"],
    "collaborative": [model for model, config in MODEL_REGISTRY.items() if config["approach"] == "collaborative"],
    "neural": [model for model, config in MODEL_REGISTRY.items() if config["approach"] == "neural"],
    "hybrid": [model for model, config in MODEL_REGISTRY.items() if config["approach"] == "hybrid"]
}

# =============================================================================
# DOUBLE DESCENT EXPERIMENT CONFIG
# =============================================================================

DOUBLE_DESCENT_CONFIG = {
    "description": "Systematically vary model complexity to observe test error curve.",
    "enabled": True,
    "output_dir": "double_descent_analysis",
    "models": {
        "content_rf": {
            "param_to_vary": "max_depth",
            "values": [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]
        },
        "collaborative_svd": {
            "param_to_vary": "n_components",
            "values": [5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500, 600]
        },
        "neural_ncf": {
            "param_to_vary": "layers",
            "values": [[32], [64], [128], [256], [512], [64, 32], [128, 64], [256, 128], [512, 256], [256, 128, 64], [512, 256, 128], [1024, 512, 256]]
        }
    }
}


# =============================================================================
# EXPERIMENTAL SETTINGS
# =============================================================================

# Data splitting settings
TEST_SIZE = 0.2  # Proportion of data for testing
RANDOM_STATE = 42  # For reproducible splits

# User group definitions based on activity level (honest analysis for MovieLens 1M)
# Note: MovieLens 1M has NO true cold start users - all users have 20+ ratings
# Thresholds adjusted for the 200k sample to ensure balanced groups
USER_GROUP_THRESHOLDS = {
    "moderate_users": 100,      # ≤100 ratings (bottom 33% - moderately active)
    "active_users": 300,        # 100-300 ratings (middle 33% - quite active)  
    "power_users": float('inf') # >300 ratings (top 33% - extremely active)
    # "moderate_users": 8,       # Bottom 33% for the 200k sample
    # "active_users": 24,        # Middle 33% for the 200k sample
    # "power_users": float('inf')  # Top 33% for the 200k sample
}

# Evaluation settings
EVALUATION_METRICS = ["precision_at_k", "recall_at_k", "ndcg_at_k", "coverage"]
TOP_K_VALUES = [5, 10, 20]  # Different k values for evaluation

# Explainability settings
EXPLAINABILITY_ENABLED = True
EXPLAINABILITY_SAMPLE_SIZE = 100  # Number of users to explain

# Results settings  
SAVE_DETAILED_RESULTS = True
SAVE_EXPLAINABILITY_DATA = True

# =============================================================================
# BACKWARD COMPATIBILITY (Legacy model names)
# =============================================================================

# Map alternative model names to main ones
MODEL_ALIASES = {
    "low_logistic": "content_logistic",
    "low_knn": "content_knn", 
    "low_svd": "collaborative_svd",
    "medium_rf": "content_rf",
    "medium_gb": "content_gb",
    "medium_fm": "hybrid_weighted",  # Will be implemented
    "medium_svdpp": "collaborative_nmf",
    "high_ncf": "neural_ncf",        # Neural Collaborative Filtering
    "high_autoencoder": "neural_autoencoder",  # Autoencoder
    "high_transformer": "neural_transformer"  # Transformer
}

# =============================================================================
# FLEXIBLE EXPERIMENT SELECTION
# =============================================================================

def get_models_for_experiment(experiment_name: str) -> List[str]:
    """Get list of models for a specific experiment."""
    if experiment_name in EXPERIMENT_CONFIGS:
        return EXPERIMENT_CONFIGS[experiment_name]["models"]
    elif experiment_name == "all":
        return list(MODEL_REGISTRY.keys())
    elif experiment_name in APPROACH_TYPES:
        return APPROACH_TYPES[experiment_name]
    elif experiment_name in COMPLEXITY_LEVELS:
        return COMPLEXITY_LEVELS[experiment_name]
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

def get_enabled_experiments() -> List[str]:
    """Get list of enabled experiments."""
    return [name for name, config in EXPERIMENT_CONFIGS.items() if config["enabled"]] 