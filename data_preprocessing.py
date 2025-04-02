import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import awswrangler as wr
from qdrant_client import QdrantClient
from typing import List, Dict, Tuple, Optional, Union
from transformers import CLIPTextModel, CLIPTokenizer
import re

class RuleBasedTransformer:
    """
    Rule-based component for query transformation that uses predefined mappings
    to convert problematic queries to safer alternatives.
    """
    def __init__(self, custom_mappings: Optional[Dict[str, str]] = None):
        # Default mappings for problematic terms
        self.term_mappings = {
            "girl": "woman",
            "sexy": "attractive",
            "hot": "appealing",
            "babe": "person",
            "young girl": "young woman",
            "naked": "portrait",
            "nude": "portrait",
            # Add more mappings as needed
        }
        
        # Add custom mappings if provided
        if custom_mappings:
            self.term_mappings.update(custom_mappings)
        
        # Compile regex patterns for efficient matching
        self.patterns = {
            re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE): v
            for k, v in self.term_mappings.items()
        }
    
    def transform(self, query: str) -> str:
        """
        Transform a query by replacing problematic terms with safer alternatives.
        
        Args:
            query: The input search query
            
        Returns:
            Transformed query with safer terms
        """
        transformed_query = query
        for pattern, replacement in self.patterns.items():
            transformed_query = pattern.sub(replacement, transformed_query)
        
        return transformed_query

class NeuralQueryTransformer(nn.Module):
    """
    Neural network-based component for query transformation that learns to
    rewrite problematic queries while preserving the core search intent.
    """
    def __init__(self, vocab_size: int = 30000, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.1):
        super(NeuralQueryTransformer, self).__init__()
        
        # Encoder-decoder architecture for query transformation
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_dim * 2,  # Bidirectional encoder output
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor, 
                target_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the neural query transformer.
        
        Args:
            input_ids: Tokenized input query
            input_mask: Attention mask for input
            target_ids: Tokenized target query for teacher forcing during training
            
        Returns:
            Logits for the transformed query
        """
        input_emb = self.embedding(input_ids)
        
        # Encode the input query
        encoder_output, (hidden, cell) = self.encoder(input_emb)
        
        # Initialize decoder input with BOS token
        batch_size = input_ids.size(0)
        decoder_input = torch.zeros_like(input_ids[:, :1])  # BOS token
        decoder_hidden = (hidden, cell)
        
        # Reshape hidden for decoder (concat forward & backward)
        hidden_reshaped = hidden.view(self.encoder.num_layers, 2, batch_size, -1)
        hidden_decoder = hidden_reshaped.transpose(1, 2).contiguous().view(
            self.encoder.num_layers, batch_size, -1)
        
        cell_reshaped = cell.view(self.encoder.num_layers, 2, batch_size, -1)
        cell_decoder = cell_reshaped.transpose(1, 2).contiguous().view(
            self.encoder.num_layers, batch_size, -1)
        
        decoder_state = (hidden_decoder, cell_decoder)
        
        # Decode step by step
        outputs = []
        max_len = 50 if target_ids is None else target_ids.size(1)
        
        for t in range(max_len):
            if target_ids is not None and t > 0 and torch.rand(1).item() < 0.5:
                # Teacher forcing with 50% probability during training
                decoder_input = target_ids[:, t-1:t]
            
            decoder_emb = self.embedding(decoder_input)
            decoder_output, decoder_state = self.decoder(decoder_emb, decoder_state)
            
            # Apply attention over encoder outputs
            attn_output, _ = self.attention(
                decoder_output, encoder_output, encoder_output, 
                key_padding_mask=(input_mask == 0)
            )
            
            # Project to vocabulary
            logits = self.output_projection(attn_output)
            outputs.append(logits)
            
            # Get next input for decoder (use predicted token)
            if target_ids is None:
                decoder_input = torch.argmax(logits, dim=-1)
        
        return torch.cat(outputs, dim=1)

class QueryTransformer:
    """
    Complete query transformation system that combines rule-based and
    neural approaches to transform problematic queries.
    """
    def __init__(self, model_path: Optional[str] = None, 
                 custom_rules: Optional[Dict[str, str]] = None):
        # Initialize the rule-based component
        self.rule_based = RuleBasedTransformer(custom_mappings=custom_rules)
        
        # Initialize the neural component
        self.neural_transformer = None
        self.tokenizer = None
        
        # Load pre-trained neural transformer if path is provided
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> None:
        """
        Load a pre-trained neural transformer model.
        
        Args:
            model_path: Path to the model weights
        """
        # In a real implementation, this would load the tokenizer and model
        # For this template, we'll just create a dummy model
        self.neural_transformer = NeuralQueryTransformer()
        # Load weights: self.neural_transformer.load_state_dict(torch.load(model_path))
        self.neural_transformer.eval()
    
    def transform_query(self, query: str, use_neural: bool = True) -> str:
        """
        Transform a query using both rule-based and optionally neural approaches.
        
        Args:
            query: The input search query
            use_neural: Whether to use the neural transformer after rule-based
            
        Returns:
            Transformed query that preserves search intent while avoiding problematic content
        """
        # First apply rule-based transformation
        transformed_query = self.rule_based.transform(query)
        
        # Apply neural transformation if available and requested
        if use_neural and self.neural_transformer is not None:
            # This would be implemented with actual tokenization and model inference
            # For this template, we're just returning the rule-based transformation
            pass
        
        return transformed_query
    
    def batch_transform(self, queries: List[str], use_neural: bool = True) -> List[str]:
        """
        Transform a batch of queries.
        
        Args:
            queries: List of input search queries
            use_neural: Whether to use the neural transformer
            
        Returns:
            List of transformed queries
        """
        return [self.transform_query(q, use_neural) for q in queries]
    
    def detect_problematic_query(self, query: str) -> bool:
        """
        Detect if a query contains problematic terms that need transformation.
        
        Args:
            query: The input search query
            
        Returns:
            True if the query contains problematic terms, False otherwise
        """
        transformed = self.rule_based.transform(query)
        return transformed != query

class CombinedVectorProcessor(nn.Module):
    """
    Process and combine image embeddings with metadata into a single vector.
    """
    def __init__(self, img_dim=1024, metadata_dim=512, output_dim=1024):
        super(CombinedVectorProcessor, self).__init__()
        
        # Process image embedding
        self.img_processor = nn.Sequential(
            nn.Linear(img_dim, img_dim),
            nn.ReLU()
        )
        
        # Process metadata embedding
        self.metadata_processor = nn.Sequential(
            nn.Linear(metadata_dim, metadata_dim),
            nn.ReLU()
        )
        
        # Combine both embeddings
        self.combiner = nn.Sequential(
            nn.Linear(img_dim + metadata_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, img_embedding, metadata_embedding):
        img_processed = self.img_processor(img_embedding)
        metadata_processed = self.metadata_processor(metadata_embedding)
        
        # Concatenate both embeddings
        combined = torch.cat([img_processed, metadata_processed], dim=-1)
        
        # Process the combined embedding
        return self.combiner(combined)

class EmbeddingManager:
    """
    Handle text and metadata embeddings separately according to the rules.
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_model = CLIPTextModel.from_pretrained(clip_model_name)
        
        # Separate embedding for metadata
        self.metadata_embedder = nn.Embedding(10000, 512)  # Simplified example
        
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Get CLIP embedding for natural language descriptions only.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.text_model(**inputs)
        return outputs.pooler_output
    
    def get_metadata_embedding(self, metadata: Dict) -> torch.Tensor:
        """
        Process metadata separately from natural language descriptions.
        """
        # Convert metadata to numerical values for embedding
        # This is a simplified example - in practice, you would need more sophisticated processing
        metadata_ids = []
        for key, value in metadata.items():
            # Hash the key-value pair to get a numerical ID
            metadata_id = hash(f"{key}:{value}") % 10000
            metadata_ids.append(metadata_id)
        
        # Convert to tensor and get embeddings
        metadata_tensor = torch.tensor(metadata_ids)
        return self.metadata_embedder(metadata_tensor).mean(dim=0)

class DataPreprocessor:
    """
    Main class for data preprocessing.
    """
    def __init__(
        self,
        qdrant_url: str = "http://localhost",
        qdrant_collection: str = "images",
        qdrant_api_key: Optional[str] = None,
        embedding_dim: int = 1024,
        metadata_dim: int = 512
    ):
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.qdrant_collection = qdrant_collection
        
        # Initialize components
        self.query_transformer = QueryTransformer()
        self.vector_processor = CombinedVectorProcessor(
            img_dim=embedding_dim, 
            metadata_dim=metadata_dim,
            output_dim=embedding_dim
        )
        self.embedding_manager = EmbeddingManager()
        
    def process_query(self, query: str) -> torch.Tensor:
        """
        Process a search query to get transformed embedding.
        """
        # Apply term replacement to avoid problematic queries
        safe_query = self.query_transformer.transform_query(query)
        
        # Get CLIP embedding for the safe query
        query_embedding = self.embedding_manager.get_text_embedding(safe_query)
        
        # Apply neural transformation to further adjust the embedding
        transformed_embedding = self.query_transformer.neural_transformer(query_embedding)
        
        return transformed_embedding
    
    def process_item(
        self, 
        image_embedding: torch.Tensor, 
        description: str, 
        metadata: Dict
    ) -> torch.Tensor:
        """
        Process an item (image + description + metadata) to get combined embedding.
        """
        # Get text embedding for natural language description
        text_embedding = self.embedding_manager.get_text_embedding(description)
        
        # Get separate embedding for metadata
        metadata_embedding = self.embedding_manager.get_metadata_embedding(metadata)
        
        # Combine image embedding with metadata embedding
        combined_embedding = self.vector_processor(image_embedding, metadata_embedding)
        
        return combined_embedding
    
    def preprocess_dataset(self, input_parquet: str, output_parquet: str):
        """
        Preprocess an entire dataset of images and metadata.
        """
        # Read the input dataset
        df = wr.s3.read_parquet(path=input_parquet)
        
        # Process each item
        processed_embeddings = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing item {idx}/{len(df)}")
            
            image_embedding = torch.tensor(row["image_embedding"])
            description = row["description"]
            metadata = row["metadata"]
            
            # Process the item
            processed_embedding = self.process_item(
                image_embedding=image_embedding,
                description=description,
                metadata=metadata
            )
            
            processed_embeddings.append(processed_embedding.detach().numpy())
        
        # Add processed embeddings to the dataframe
        df["processed_embedding"] = processed_embeddings
        
        # Save the preprocessed dataset
        wr.s3.to_parquet(df=df, path=output_parquet)
        
        print(f"Preprocessed dataset saved to {output_parquet}")
    
    def update_qdrant_collection(self, input_parquet: str):
        """
        Update Qdrant collection with preprocessed embeddings.
        """
        df = wr.s3.read_parquet(path=input_parquet)
        
        points = []
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Preparing point {idx}/{len(df)} for Qdrant")
            
            point = {
                "id": row["id"],
                "vector": row["processed_embedding"],
                "payload": {
                    "description": row["description"],
                    "metadata": row["metadata"]
                }
            }
            points.append(point)
        
        # Insert points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.qdrant_client.upsert(
                collection_name=self.qdrant_collection,
                points=batch
            )
        
        print(f"Updated Qdrant collection with {len(points)} points")

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for safer and more effective search")
    
    parser.add_argument("--input-parquet", type=str, required=True, 
                        help="Path to input parquet file containing raw data")
    parser.add_argument("--output-parquet", type=str, required=True,
                        help="Path to save preprocessed data")
    parser.add_argument("--update-qdrant", action="store_true",
                        help="Update Qdrant collection with preprocessed embeddings")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost",
                        help="URL of Qdrant service")
    parser.add_argument("--qdrant-collection", type=str, default="images",
                        help="Name of Qdrant collection")
    parser.add_argument("--qdrant-api-key", type=str, default=None,
                        help="API key for Qdrant service")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
        qdrant_api_key=args.qdrant_api_key
    )
    
    # Preprocess dataset
    preprocessor.preprocess_dataset(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet
    )
    
    # Update Qdrant collection if requested
    if args.update_qdrant:
        preprocessor.update_qdrant_collection(input_parquet=args.output_parquet)

if __name__ == "__main__":
    main() 