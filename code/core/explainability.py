"""
Explainability module for recommendation models.
Implements SHAP and LIME-based explanations for thesis research.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Try to import explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

try:
    from lime import lime_tabular
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")

class RecommendationExplainer:
    """
    Provides explainability for recommendation models using SHAP and LIME.
    Essential for thesis research on complexity vs explainability trade-offs.
    """
    
    def __init__(
        self,
        feature_names: List[str],
        feature_values: np.ndarray,
        model: object,
        sample_size: int = 500
    ):
        """
        Initialize explainer.
        
        Args:
            feature_names: Names of features
            feature_values: Training feature matrix
            model: Fitted recommendation model
            sample_size: Sample size for background data
        """
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.model = model
        self.sample_size = min(sample_size, len(feature_values))
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        if SHAP_AVAILABLE:
            self.shap_explainer = self._initialize_shap()
        else:
            logger.warning("SHAP not available - skipping SHAP initialization")
            
        if LIME_AVAILABLE:
            self.lime_explainer = self._initialize_lime()
        else:
            logger.warning("LIME not available - skipping LIME initialization")
    
    def _predict_wrapper(self, X):
        """Wrapper function that ensures consistent output for SHAP/LIME."""
        try:
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(X)
            elif hasattr(self.model, 'similarity_matrix'):
                # For similarity-based models, return average similarity
                if X.shape[0] == 1:
                    idx = 0  # Single prediction
                    similarities = self.model.similarity_matrix[idx]
                    predictions = np.array([[np.mean(similarities)]])
                else:
                    predictions = np.array([[0.5] for _ in range(len(X))])
            else:
                # Fallback: return neutral predictions
                predictions = np.array([[0.5] for _ in range(len(X))])
            
            # Ensure correct shape
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            elif predictions.ndim > 2:
                predictions = predictions.reshape(len(predictions), -1)
            
            # If multi-output, take first column or average
            if predictions.shape[1] > 1:
                predictions = predictions.mean(axis=1, keepdims=True)
                
            return predictions.flatten()
            
        except Exception as e:
            logger.warning(f"Prediction wrapper failed: {e}")
            return np.array([0.5] * len(X))
    
    def _initialize_shap(self) -> Optional[object]:
        """Initialize SHAP explainer with KernelExplainer for better compatibility."""
        try:
            # Sample data for SHAP background
            sample_indices = np.random.choice(
                len(self.feature_values), 
                size=self.sample_size, 
                replace=False
            )
            background_data = self.feature_values[sample_indices]
            
            # Initialize KernelExplainer (model-agnostic)
            explainer = shap.KernelExplainer(
                self._predict_wrapper,
                background_data,
                link="identity"
            )
            
            logger.info(f"Initialized SHAP explainer with {len(background_data)} background samples")
            return explainer
            
        except Exception as e:
            logger.warning(f"SHAP initialization failed: {e}")
            return None
    
    def _initialize_lime(self) -> Optional[LimeTabularExplainer]:
        """Initialize LIME explainer."""
        try:
            return LimeTabularExplainer(
                self.feature_values,
                feature_names=self.feature_names,
                mode='regression',
                discretize_continuous=True,
                random_state=42
            )
        except Exception as e:
            logger.warning(f"LIME initialization failed: {e}")
            return None
    
    def get_shap_explanation(
        self,
        item_idx: int,
        n_features: int = 10
    ) -> Dict[str, float]:
        """
        Get SHAP explanation for an item.
        
        Args:
            item_idx: Index of item to explain
            n_features: Number of top features to return
            
        Returns:
            Dictionary of feature names and their SHAP values
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        try:
            # Get item features
            item_features = self.feature_values[item_idx].reshape(1, -1)
            
            # Calculate SHAP values with reduced sample size for speed
            shap_values = self.shap_explainer.shap_values(
                item_features,
                nsamples=min(100, self.sample_size // 5)  # Reduce computational load
            )
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-output case, take first output
                values = shap_values[0]
            else:
                values = shap_values
            
            # Ensure we have the right shape
            if values.ndim > 1:
                values = values.flatten()
            
            # Get top features by absolute SHAP value
            abs_values = np.abs(values)
            top_indices = np.argsort(abs_values)[::-1][:n_features]
            
            # Create explanation dictionary
            explanation = {}
            for idx in top_indices:
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    shap_value = float(values[idx])
                    explanation[feature_name] = shap_value
            
            return explanation
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            # Fallback: return simple feature importance
            return self._get_fallback_explanation(item_idx, n_features)
    
    def get_lime_explanation(
        self,
        item_idx: int,
        n_features: int = 10
    ) -> Dict[str, float]:
        """
        Get LIME explanation for an item.
        
        Args:
            item_idx: Index of item to explain
            n_features: Number of top features to return
            
        Returns:
            Dictionary of feature names and their LIME values
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized")
        
        try:
            # Get item features
            item_features = self.feature_values[item_idx]
            
            # Generate LIME explanation
            exp = self.lime_explainer.explain_instance(
                item_features,
                self._predict_wrapper,
                num_features=n_features,
                num_samples=min(500, self.sample_size)  # Reduce for speed
            )
            
            # Extract explanation
            explanation = {}
            
            # Handle different LIME output formats
            for feature_idx, importance in exp.as_list():
                # Extract feature name from LIME description (e.g., "feature_name <= 0.5")
                if isinstance(feature_idx, str):
                    # Parse feature name from condition string
                    feature_name = feature_idx.split()[0]
                elif isinstance(feature_idx, int):
                    # Direct feature index
                    if feature_idx < len(self.feature_names):
                        feature_name = self.feature_names[feature_idx]
                    else:
                        feature_name = f"feature_{feature_idx}"
                else:
                    feature_name = str(feature_idx)
                
                explanation[feature_name] = float(importance)
            
            return explanation
            
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            # Fallback: return simple feature importance
            return self._get_fallback_explanation(item_idx, n_features)
    
    def _get_fallback_explanation(self, item_idx: int, n_features: int) -> Dict[str, float]:
        """Fallback explanation when SHAP/LIME fail."""
        try:
            # Simple feature importance based on feature values
            item_features = self.feature_values[item_idx]
            
            # Get features with highest absolute values
            abs_values = np.abs(item_features)
            top_indices = np.argsort(abs_values)[::-1][:n_features]
            
            explanation = {}
            for idx in top_indices:
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    value = float(item_features[idx])
                    explanation[feature_name] = value
            
            return explanation
            
        except Exception as e:
            logger.warning(f"Fallback explanation failed: {e}")
            return {"error": "explanation_failed"}

def explain_recommendation(
    item_idx: int,
    recommender: object,
    feature_names: List[str],
    feature_values: np.ndarray,
    method: str = "shap",
    n_features: int = 10
) -> Dict[str, float]:
    """
    Explain a recommendation using SHAP or LIME.
    
    Args:
        item_idx: Index of item to explain
        recommender: Fitted recommender model
        feature_names: List of feature names
        feature_values: Feature matrix
        method: Either "shap" or "lime"
        n_features: Number of features to include in explanation
        
    Returns:
        Dictionary of feature importances
    """
    try:
        explainer = RecommendationExplainer(
            feature_names, feature_values, recommender
        )
        
        if method == "shap":
            return explainer.get_shap_explanation(item_idx, n_features)
        elif method == "lime":
            return explainer.get_lime_explanation(item_idx, n_features)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        return {"error": str(e)}

def analyze_feature_importance_across_models(
    models: Dict[str, object],
    feature_names: List[str],
    feature_values: np.ndarray,
    sample_items: List[int],
    methods: List[str] = ["shap", "lime"]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Analyze feature importance across different models for thesis research.
    
    Args:
        models: Dictionary of model_name -> fitted_model
        feature_names: List of feature names
        feature_values: Feature matrix
        sample_items: List of item indices to analyze
        methods: Explanation methods to use
        
    Returns:
        Nested dictionary: {model_name: {method: {feature: importance}}}
    """
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Analyzing feature importance for {model_name}")
        results[model_name] = {}
        
        for method in methods:
            logger.info(f"  Using method: {method}")
            method_results = {}
            
            for item_idx in sample_items:
                try:
                    explanation = explain_recommendation(
                        item_idx, model, feature_names, feature_values, method
                    )
                    
                    # Aggregate feature importance across items
                    for feature, importance in explanation.items():
                        if feature not in method_results:
                            method_results[feature] = []
                        method_results[feature].append(importance)
                        
                except Exception as e:
                    logger.warning(f"Failed to explain item {item_idx} for {model_name}: {e}")
            
            # Average importance across items
            averaged_results = {}
            for feature, importances in method_results.items():
                averaged_results[feature] = np.mean(importances)
            
            results[model_name][method] = averaged_results
    
    return results

def generate_explainability_report(
    explainability_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str = "explainability_report.html"
) -> str:
    """
    Generate HTML report comparing explainability across models.
    
    Args:
        explainability_results: Results from analyze_feature_importance_across_models
        output_path: Path to save HTML report
        
    Returns:
        Path to generated report
    """
    html_content = """
    <html>
    <head>
        <title>Model Explainability Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .model-section { margin: 20px 0; border: 1px solid #ddd; padding: 20px; }
            .method-section { margin: 10px 0; }
            table { border-collapse: collapse; width: 100%; margin: 10px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .positive { color: green; }
            .negative { color: red; }
        </style>
    </head>
    <body>
        <h1>Model Explainability Analysis</h1>
        <p>Comparison of feature importance across different recommendation models.</p>
    """
    
    for model_name, model_results in explainability_results.items():
        html_content += f"""
        <div class="model-section">
            <h2>Model: {model_name}</h2>
        """
        
        for method, feature_importance in model_results.items():
            html_content += f"""
            <div class="method-section">
                <h3>Method: {method.upper()}</h3>
                <table>
                    <tr><th>Feature</th><th>Importance</th></tr>
            """
            
            # Sort features by absolute importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]  # Top 10 features
            
            for feature, importance in sorted_features:
                color_class = "positive" if importance > 0 else "negative"
                html_content += f"""
                    <tr>
                        <td>{feature}</td>
                        <td class="{color_class}">{importance:.4f}</td>
                    </tr>
                """
            
            html_content += "</table></div>"
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Explainability report saved to: {output_path}")
    return output_path 