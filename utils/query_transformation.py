import torch
import torch.nn as nn
import numpy as np
import re
from typing import Dict, List, Optional

class QueryTransformer:
    """
    Class for transforming and preprocessing search queries.
    Handles problematic terms and applies transformations to improve search results.
    """
    def __init__(self, embedding_dim: int = 1024, hidden_dim: int = 512):
        """
        Initialize the query transformer.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            hidden_dim: Dimension of the hidden layers in the transformation network
        """
        # Neural network for transforming query embeddings
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        
        # Dictionary of problematic terms and their safe replacements
        self.term_replacements = {
            "girl": "jubilant energetic women",
            "teen": "young adult",
            "sexy": "professional attire",
            "hot": "vibrant",
            "nude": "natural portrait",
            "bikini": "beachwear",
            "lingerie": "loungewear",
            "young woman": "adult woman",
            # Add more terms as needed
        }
        
        # Compile regex patterns for more efficient replacement
        self.patterns = {re.compile(rf'\b{re.escape(k)}\b', re.IGNORECASE): v 
                         for k, v in self.term_replacements.items()}
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query by replacing problematic terms with safer alternatives.
        
        Args:
            query: The search query
            
        Returns:
            Preprocessed query with problematic terms replaced
        """
        if not query:
            return query
        
        # Apply all replacements
        processed_query = query
        for pattern, replacement in self.patterns.items():
            processed_query = pattern.sub(replacement, processed_query)
        
        return processed_query
    
    def transform_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Apply neural transformation to query embedding.
        
        Args:
            embedding: Query embedding tensor
            
        Returns:
            Transformed embedding
        """
        with torch.no_grad():
            return self.transform(embedding)
    
    def process_query(self, query: str, embedding_model) -> np.ndarray:
        """
        Complete query processing pipeline: text preprocessing and embedding transformation.
        
        Args:
            query: The search query
            embedding_model: Model to convert text to embeddings
            
        Returns:
            Transformed query embedding
        """
        # Text preprocessing
        safe_query = self.preprocess_query(query)
        
        # Embed the preprocessed query
        query_embedding = embedding_model.encode(safe_query)
        
        # Convert to tensor
        if not isinstance(query_embedding, torch.Tensor):
            query_embedding = torch.tensor(query_embedding)
        
        # Neural transformation
        transformed_embedding = self.transform_embedding(query_embedding)
        
        # Convert back to numpy array
        return transformed_embedding.numpy()

# Test the query transformer
if __name__ == "__main__":
    transformer = QueryTransformer()
    test_queries = [
        "girl in a park",
        "sexy dress fashion",
        "teen activities",
        "hot summer day",
        "mountain landscape",
        "cityscape at night"
    ]
    
    print("Query Transformation Examples:")
    print("-" * 50)
    for query in test_queries:
        transformed = transformer.preprocess_query(query)
        print(f"Original: '{query}'")
        print(f"Transformed: '{transformed}'")
        print("-" * 50) 