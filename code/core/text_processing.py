"""
Handles text feature enrichment using pre-trained language models (BERT).
This module is responsible for fetching movie synopses, generating embeddings,
and saving them for later use by the DataManager.
"""
import pandas as pd
import numpy as np
import os
import pickle
import logging
from typing import Dict

# Ensure transformers and torch are installed
try:
    import torch
    from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TextFeatureEnricher:
    """
    Enriches movie data with text embeddings from synopses.
    """
    def __init__(self, model_name='distilbert-base-uncased', cache_path='../datasets'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not found. Cannot proceed with text enrichment.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Using DistilBERT for a good balance of performance and size
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)
        self.embedding_cache_path = os.path.join(cache_path, 'movie_embeddings.pkl')

    def _fetch_movie_synopses(self, movies_df: pd.DataFrame) -> Dict[int, str]:
        """
        Placeholder for fetching movie synopses from an external API (e.g., OMDb).
        In a real scenario, this would involve API calls. Here, we'll simulate it.
        """
        logger.info("Simulating fetching movie synopses...")
        synopses = {}
        for _, row in movies_df.iterrows():
            movie_id = row['movie_id']
            title = row['title']
            # Simple simulation: create a "synopsis" from the title and genres
            synopsis = f"A movie titled {title}. Genres include {row.get('genres', 'various')}."
            synopses[movie_id] = synopsis
        
        return synopses

    def generate_embeddings(self, text_dict: Dict[int, str], batch_size=32) -> Dict[int, np.ndarray]:
        """
        Generates embeddings for a dictionary of texts (movie_id -> synopsis).
        """
        logger.info(f"Generating embeddings for {len(text_dict)} texts...")
        self.model.eval()
        
        movie_ids = list(text_dict.keys())
        texts = list(text_dict.values())
        all_embeddings = {}

        for i in range(0, len(texts), batch_size):
            batch_ids = movie_ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use the [CLS] token embedding as the representation of the sequence
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            for movie_id, embedding in zip(batch_ids, batch_embeddings):
                all_embeddings[movie_id] = embedding
        
        logger.info("Embeddings generated successfully.")
        return all_embeddings

    def process_and_save_embeddings(self, movies_df: pd.DataFrame):
        """
        Main orchestration method. Fetches synopses, generates embeddings, and saves them to a file.
        """
        if os.path.exists(self.embedding_cache_path):
            logger.info(f"Embeddings already exist at {self.embedding_cache_path}. Skipping generation.")
            return

        synopses = self._fetch_movie_synopses(movies_df)
        embeddings = self.generate_embeddings(synopses)

        with open(self.embedding_cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        logger.info(f"Movie embeddings saved to {self.embedding_cache_path}")

    def load_embeddings(self) -> Dict[int, np.ndarray]:
        """
        Loads pre-computed embeddings from cache.
        """
        if not os.path.exists(self.embedding_cache_path):
            logger.error("Embeddings cache not found. Please run process_and_save_embeddings first.")
            return {}
        
        with open(self.embedding_cache_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        logger.info(f"Loaded {len(embeddings)} movie embeddings from cache.")
        return embeddings

if __name__ == '__main__':
    # Example usage:
    # This would be called once as a preprocessing step.
    
    # Load your main movie data
    # movies_df = pd.read_csv('../datasets/movies.csv') # Adjust path as needed
    
    # enricher = TextFeatureEnricher()
    # enricher.process_and_save_embeddings(movies_df)
    pass 