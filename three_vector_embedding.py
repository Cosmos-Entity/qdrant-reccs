import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import awswrangler as wr
from qdrant_client import QdrantClient
from typing import List, Dict, Tuple, Optional, Union, Any
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModel
import openai
import os
import json
from sklearn.preprocessing import normalize

class LLMCaptionGenerator:
    """
    Generate captions for images using LLM.
    """
    def __init__(self, api_key=None, model="gpt-4o"):
        if api_key:
            openai.api_key = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            openai.api_key = os.environ.get("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided or set as environment variable")
            
        self.model = model
        self.prompt_template = """
        You are an expert Cosmos reviewer. Please provide a detailed description of the image.
        
        Consider the following aspects:
        - Main subject and visual content
        - Style and composition
        - Colors and aesthetics
        - Emotional impact
        - Technical quality (focus, lighting)
        
        Please rate the following on a scale of 1-10:
        - Composition: 
        - Technical Quality: 
        - Artistic Value: 
        - Originality: 
        
        If the image contains any inappropriate content (nudity, violence, offensive material),
        indicate this with a content warning.
        
        Provide your full analysis in 3-5 sentences.
        """
    
    def generate_caption(self, image_url: str) -> str:
        """
        Generate a caption for the image using the LLM.
        
        In a real implementation, this would send the image to OpenAI's API.
        For this example, we'll simulate by returning a placeholder.
        """
        # In a real implementation, this would call the OpenAI API with the image
        # response = openai.ChatCompletion.create(
        #     model=self.model,
        #     messages=[
        #         {"role": "system", "content": self.prompt_template},
        #         {"role": "user", "content": [
        #             {"type": "image_url", "image_url": {"url": image_url}}
        #         ]}
        #     ]
        # )
        # return response.choices[0].message.content
        
        # For this example, we'll return a placeholder caption
        return "A high-quality professionally composed image with warm tones. The composition is balanced with careful attention to lighting. Composition: 8, Technical Quality: 9, Artistic Value: 7, Originality: 6."

class TextEmbeddingModel:
    """
    Generate text embeddings using a model other than CLIP.
    """
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Generate an embedding for the text.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean pooling of the last hidden state as the embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze()

class MetadataProcessor:
    """
    Process metadata into a standardized format for embedding.
    """
    def __init__(self):
        pass
    ``
    def process_metadata(self, metadata: Dict) -> str:
        """
        Convert metadata dictionary to a string for embedding.
        """
        metadata_str = ""
        
        # Process different metadata fields
        if "tags" in metadata:
            metadata_str += f"Tags: {', '.join(metadata['tags'])}. "
        
        if "category" in metadata:
            metadata_str += f"Category: {metadata['category']}. "
            
        if "author" in metadata:
            metadata_str += f"Author: {metadata['author']}. "
            
        if "date" in metadata:
            metadata_str += f"Date: {metadata['date']}. "
            
        if "location" in metadata:
            metadata_str += f"Location: {metadata['location']}. "
            
        if "device" in metadata:
            metadata_str += f"Device: {metadata['device']}. "
            
        if "settings" in metadata:
            settings = metadata["settings"]
            settings_str = ", ".join([f"{k}: {v}" for k, v in settings.items()])
            metadata_str += f"Settings: {settings_str}. "
        
        return metadata_str.strip()

class ThreeVectorCombiner(nn.Module):
    """
    Combine three vectors into a single vector with learnable weights.
    """
    def __init__(self, vector_dim=1024, learn_weights=True):
        super(ThreeVectorCombiner, self).__init__()
        self.vector_dim = vector_dim
        self.learn_weights = learn_weights
        
        # Initialize weights - these will be learned during training
        if learn_weights:
            self.image_weight = nn.Parameter(torch.tensor(0.33))
            self.text_weight = nn.Parameter(torch.tensor(0.33))
            self.metadata_weight = nn.Parameter(torch.tensor(0.33))
        else:
            # Fixed weights
            self.register_buffer('image_weight', torch.tensor(0.33))
            self.register_buffer('text_weight', torch.tensor(0.33))
            self.register_buffer('metadata_weight', torch.tensor(0.33))
        
        # Optional: transformation after combining
        self.transform = nn.Sequential(
            nn.Linear(vector_dim, vector_dim),
            nn.LayerNorm(vector_dim)
        )
    
    def forward(self, image_vector, text_vector, metadata_vector):
        """
        Combine the three vectors using learned weights.
        """
        # Ensure all vectors have the same dimension
        if image_vector.shape != text_vector.shape or image_vector.shape != metadata_vector.shape:
            raise ValueError("All vectors must have the same dimension")
        
        # Normalize weights to sum to 1
        total = self.image_weight + self.text_weight + self.metadata_weight
        image_weight_norm = self.image_weight / total
        text_weight_norm = self.text_weight / total
        metadata_weight_norm = self.metadata_weight / total
        
        # Combine vectors using weights
        combined = (
            image_weight_norm * image_vector +
            text_weight_norm * text_vector +
            metadata_weight_norm * metadata_vector
        )
        
        # Apply transformation
        return self.transform(combined)
    
    def get_weights(self):
        """
        Return the current weights (normalized).
        """
        total = self.image_weight + self.text_weight + self.metadata_weight
        return {
            "image_weight": (self.image_weight / total).item(),
            "text_weight": (self.text_weight / total).item(),
            "metadata_weight": (self.metadata_weight / total).item()
        }

class ImageEncoder(nn.Module):
    """
    Wrapper for image encoding, typically using a CLIP model.
    """
    def __init__(self, pretrained_model_name: str = "openai/clip-vit-base-patch32"):
        super(ImageEncoder, self).__init__()
        self.embedding_dim = 512  # CLIP's default embedding dimension
        
        # In a real implementation, this would load the CLIP model
        # For this template, we'll simulate the encoder with a dummy layer
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),  # Simulate image feature extraction
            nn.ReLU(),
            nn.Linear(1024, self.embedding_dim)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: Batch of images
            
        Returns:
            Image embeddings
        """
        return self.encoder(images)

class TextEncoder(nn.Module):
    """
    Encoder for text descriptions, captions, or other textual content.
    """
    def __init__(self, vocab_size: int = 30000, embedding_dim: int = 512, 
                 hidden_dim: int = 768, num_layers: int = 4):
        super(TextEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Text encoding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Output projection to match embedding dimension
        self.projection = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, text_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text_ids: Tokenized text input
            attention_mask: Mask for padded tokens
            
        Returns:
            Text embeddings
        """
        embeddings = self.embedding(text_ids)
        
        # Apply transformer with masking
        if attention_mask is not None:
            transformer_output = self.transformer(
                embeddings, 
                src_key_padding_mask=(attention_mask == 0)
            )
        else:
            transformer_output = self.transformer(embeddings)
        
        # Global pooling (mean of non-padded tokens)
        if attention_mask is not None:
            # Apply mask
            masked_output = transformer_output * attention_mask.unsqueeze(-1)
            # Sum and divide by number of tokens
            text_embedding = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            # Simple mean pooling if no mask
            text_embedding = transformer_output.mean(dim=1)
        
        # Project to final embedding space
        return self.projection(text_embedding)

class MetadataEncoder(nn.Module):
    """
    Encoder for structured metadata (categories, tags, attributes, etc.)
    """
    def __init__(self, metadata_dims: Dict[str, int], embedding_dim: int = 512):
        super(MetadataEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.metadata_dims = metadata_dims
        
        # Create embeddings for each categorical metadata field
        self.categorical_embeddings = nn.ModuleDict({
            field: nn.Embedding(dim, embedding_dim // 4)
            for field, dim in metadata_dims.items()
            if field.startswith('categorical_')
        })
        
        # Create encoders for numerical metadata fields
        self.numerical_encoders = nn.ModuleDict({
            field: nn.Sequential(
                nn.Linear(1, embedding_dim // 8),
                nn.ReLU()
            )
            for field in metadata_dims
            if field.startswith('numerical_')
        })
        
        # Create encoders for text metadata fields (like tags)
        self.text_encoders = nn.ModuleDict({
            field: nn.EmbeddingBag(dim, embedding_dim // 4, mode='mean')
            for field, dim in metadata_dims.items()
            if field.startswith('text_')
        })
        
        # Determine the total intermediate dimension
        total_dim = sum(
            [embedding_dim // 4 for f in self.categorical_embeddings] + 
            [embedding_dim // 8 for f in self.numerical_encoders] + 
            [embedding_dim // 4 for f in self.text_encoders]
        )
        
        # Final projection to the embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(total_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Field importance weights (learned)
        self.field_weights = nn.Parameter(torch.ones(len(metadata_dims)))
    
    def forward(self, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode metadata to embeddings.
        
        Args:
            metadata: Dictionary of metadata fields and their values
            
        Returns:
            Metadata embeddings
        """
        field_embeddings = []
        field_idx = 0
        
        # Process categorical fields
        for field, embedding_layer in self.categorical_embeddings.items():
            if field in metadata:
                field_emb = embedding_layer(metadata[field])
                field_emb = field_emb * self.field_weights[field_idx].unsqueeze(0).unsqueeze(-1)
                field_embeddings.append(field_emb)
            field_idx += 1
        
        # Process numerical fields
        for field, encoder in self.numerical_encoders.items():
            if field in metadata:
                # Ensure proper shape for numerical values
                values = metadata[field].float().unsqueeze(-1)
                field_emb = encoder(values)
                field_emb = field_emb * self.field_weights[field_idx].unsqueeze(0).unsqueeze(-1)
                field_embeddings.append(field_emb)
            field_idx += 1
        
        # Process text fields (e.g., tags)
        for field, encoder in self.text_encoders.items():
            if field in metadata:
                values = metadata[field]
                offsets = metadata.get(f"{field}_offsets", None)
                if offsets is not None:
                    field_emb = encoder(values, offsets)
                else:
                    # Fallback to standard embedding if offsets not provided
                    field_emb = encoder(values)
                field_emb = field_emb * self.field_weights[field_idx].unsqueeze(0).unsqueeze(-1)
                field_embeddings.append(field_emb)
            field_idx += 1
        
        # Concatenate all field embeddings
        if field_embeddings:
            combined = torch.cat(field_embeddings, dim=-1)
            # Project to final embedding space
            return self.projection(combined)
        else:
            # Return zero embedding if no metadata provided
            batch_size = next(iter(metadata.values())).size(0) if metadata else 1
            return torch.zeros(batch_size, self.embedding_dim, device=self.field_weights.device)

class DynamicWeightingLayer(nn.Module):
    """
    Layer that learns to dynamically weight the contribution of different vector types
    based on the query context.
    """
    def __init__(self, embedding_dim: int = 512):
        super(DynamicWeightingLayer, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Context-aware weighting network
        self.weighting_network = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor, 
                metadata_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamically weighted combination of the three vector types.
        
        Args:
            image_emb: Image embeddings
            text_emb: Text description embeddings
            metadata_emb: Metadata embeddings
            
        Returns:
            Weighted combination of the three embeddings
        """
        # Concatenate embeddings to get context
        concat_emb = torch.cat([image_emb, text_emb, metadata_emb], dim=-1)
        
        # Compute weights based on context
        weights = self.weighting_network(concat_emb)
        
        # Apply weights to each embedding type
        weighted_sum = (
            weights[:, 0:1] * image_emb +
            weights[:, 1:2] * text_emb +
            weights[:, 2:3] * metadata_emb
        )
        
        return weighted_sum

class ThreeVectorEncoder(nn.Module):
    """
    Complete encoder that combines image, text description, and metadata
    into a single unified vector representation.
    """
    def __init__(self, embedding_dim: int = 512, 
                 metadata_dims: Optional[Dict[str, int]] = None):
        super(ThreeVectorEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Set default metadata dimensions if not provided
        if metadata_dims is None:
            metadata_dims = {
                'categorical_category': 1000,  # Category ID
                'categorical_subcategory': 5000,  # Subcategory ID
                'numerical_popularity': 1,  # Popularity score
                'text_tags': 10000,  # Tags vocabulary size
            }
        
        # Initialize the three encoders
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(embedding_dim=embedding_dim)
        self.metadata_encoder = MetadataEncoder(metadata_dims, embedding_dim=embedding_dim)
        
        # Initialize the dynamic weighting layer
        self.weighting_layer = DynamicWeightingLayer(embedding_dim=embedding_dim)
        
        # Add projection layers to ensure all encoders output the same dimension
        self.image_projection = nn.Linear(self.image_encoder.embedding_dim, embedding_dim)
        self.text_projection = nn.Linear(self.text_encoder.embedding_dim, embedding_dim)
        self.metadata_projection = nn.Linear(self.metadata_encoder.embedding_dim, embedding_dim)
        
        # Normalization layers
        self.image_norm = nn.LayerNorm(embedding_dim)
        self.text_norm = nn.LayerNorm(embedding_dim)
        self.metadata_norm = nn.LayerNorm(embedding_dim)
        
        # Final output normalization
        self.output_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, images: Optional[torch.Tensor] = None, 
                text_descriptions: Optional[Dict[str, torch.Tensor]] = None,
                metadata: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Encode the three vector types and combine them with learned weights.
        
        Args:
            images: Batch of images
            text_descriptions: Dictionary with text_ids and attention_mask
            metadata: Dictionary of metadata fields and their values
            
        Returns:
            Combined vector representation
        """
        batch_size = images.size(0) if images is not None else (
            text_descriptions['text_ids'].size(0) if text_descriptions is not None else 
            next(iter(metadata.values())).size(0) if metadata is not None else 1
        )
        device = (images.device if images is not None else 
                 text_descriptions['text_ids'].device if text_descriptions is not None else
                 next(iter(metadata.values())).device if metadata is not None else 
                 self.image_projection.weight.device)
        
        # Process image if provided
        if images is not None:
            image_emb = self.image_encoder(images)
            image_emb = self.image_projection(image_emb)
            image_emb = self.image_norm(image_emb)
        else:
            image_emb = torch.zeros(batch_size, self.embedding_dim, device=device)
        
        # Process text description if provided
        if text_descriptions is not None:
            text_emb = self.text_encoder(
                text_descriptions['text_ids'],
                text_descriptions.get('attention_mask', None)
            )
            text_emb = self.text_projection(text_emb)
            text_emb = self.text_norm(text_emb)
        else:
            text_emb = torch.zeros(batch_size, self.embedding_dim, device=device)
        
        # Process metadata if provided
        if metadata is not None:
            metadata_emb = self.metadata_encoder(metadata)
            metadata_emb = self.metadata_projection(metadata_emb)
            metadata_emb = self.metadata_norm(metadata_emb)
        else:
            metadata_emb = torch.zeros(batch_size, self.embedding_dim, device=device)
        
        # Combine the three embeddings with dynamic weighting
        combined_emb = self.weighting_layer(image_emb, text_emb, metadata_emb)
        
        # Apply final normalization
        combined_emb = self.output_norm(combined_emb)
        
        return combined_emb
    
    def encode_query(self, query_image: Optional[torch.Tensor] = None,
                    query_text: Optional[Dict[str, torch.Tensor]] = None,
                    query_metadata: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Encode a query using the three-vector architecture.
        This is a convenience method that wraps the forward pass.
        
        Args:
            query_image: Query image
            query_text: Query text
            query_metadata: Query metadata
            
        Returns:
            Query embedding
        """
        return self.forward(query_image, query_text, query_metadata)

class ThreeVectorEmbedding:
    """
    Main class for implementing the three-vector architecture.
    """
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        text_model_name="sentence-transformers/all-mpnet-base-v2",
        openai_api_key=None,
        fixed_weights=False,
        vector_dim=1024,
        qdrant_url="http://localhost",
        qdrant_collection="images",
        qdrant_api_key=None
    ):
        # Initialize CLIP model for image vectors
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        
        # Initialize text embedding model
        self.text_embedder = TextEmbeddingModel(model_name=text_model_name)
        
        # Initialize caption generator
        self.caption_generator = LLMCaptionGenerator(api_key=openai_api_key)
        
        # Initialize metadata processor
        self.metadata_processor = MetadataProcessor()
        
        # Initialize vector combiner
        self.vector_combiner = ThreeVectorCombiner(
            vector_dim=vector_dim,
            learn_weights=not fixed_weights
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.qdrant_collection = qdrant_collection
    
    def get_image_vector(self, image_path=None, image_url=None, image_tensor=None):
        """
        Get CLIP embedding for an image.
        """
        if image_tensor is not None:
            # Use provided tensor directly
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(pixel_values=image_tensor)
            return image_features.squeeze()
        
        elif image_path is not None:
            # Load image from path
            from PIL import Image
            image = Image.open(image_path)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            return image_features.squeeze()
        
        elif image_url is not None:
            # This would download and process the image
            # For simplicity, return a dummy vector
            return torch.randn(1024)
        
        else:
            raise ValueError("Must provide one of: image_path, image_url, or image_tensor")
    
    def get_text_vector(self, description=None, image_path=None, image_url=None):
        """
        Get text embedding for a description, generating one if not provided.
        """
        if description is None:
            if image_url:
                description = self.caption_generator.generate_caption(image_url)
            elif image_path:
                # In a real implementation, you'd need to host this image somewhere
                # For this example, we'll just generate a placeholder caption
                description = self.caption_generator.generate_caption("placeholder")
            else:
                raise ValueError("Must provide either description or image_path/image_url")
        
        # Get text embedding
        return self.text_embedder.get_embedding(description)
    
    def get_metadata_vector(self, metadata):
        """
        Get embedding for metadata.
        """
        # Process metadata to string
        metadata_str = self.metadata_processor.process_metadata(metadata)
        
        # Get embedding
        return self.text_embedder.get_embedding(metadata_str)
    
    def get_combined_vector(self, image_vector, text_vector, metadata_vector):
        """
        Combine the three vectors into a single vector.
        """
        # Make sure all vectors have the same dimension
        if hasattr(image_vector, 'shape') and hasattr(text_vector, 'shape') and hasattr(metadata_vector, 'shape'):
            if image_vector.shape != text_vector.shape or image_vector.shape != metadata_vector.shape:
                # Resize vectors if necessary
                if hasattr(text_vector, 'shape') and text_vector.shape[0] != image_vector.shape[0]:
                    text_vector = nn.functional.interpolate(
                        text_vector.unsqueeze(0).unsqueeze(0), 
                        size=image_vector.shape[0]
                    ).squeeze()
                if hasattr(metadata_vector, 'shape') and metadata_vector.shape[0] != image_vector.shape[0]:
                    metadata_vector = nn.functional.interpolate(
                        metadata_vector.unsqueeze(0).unsqueeze(0), 
                        size=image_vector.shape[0]
                    ).squeeze()
        
        # Combine vectors
        return self.vector_combiner(image_vector, text_vector, metadata_vector)
    
    def process_item(self, item):
        """
        Process a single item to get all three vectors and the combined vector.
        
        Item can be a dictionary with image_path/image_url, description, and metadata.
        """
        # Get image vector
        if "image_path" in item:
            image_vector = self.get_image_vector(image_path=item["image_path"])
        elif "image_url" in item:
            image_vector = self.get_image_vector(image_url=item["image_url"])
        elif "image_tensor" in item:
            image_vector = self.get_image_vector(image_tensor=item["image_tensor"])
        else:
            raise ValueError("Item must contain image_path, image_url, or image_tensor")
        
        # Get text vector
        if "description" in item:
            text_vector = self.get_text_vector(description=item["description"])
        elif "image_path" in item:
            text_vector = self.get_text_vector(image_path=item["image_path"])
        elif "image_url" in item:
            text_vector = self.get_text_vector(image_url=item["image_url"])
        else:
            raise ValueError("Item must contain description, image_path, or image_url")
        
        # Get metadata vector
        if "metadata" in item:
            metadata_vector = self.get_metadata_vector(item["metadata"])
        else:
            # Create empty metadata if not provided
            metadata_vector = self.get_metadata_vector({})
        
        # Get combined vector
        combined_vector = self.get_combined_vector(image_vector, text_vector, metadata_vector)
        
        return {
            "image_vector": image_vector.detach().cpu().numpy(),
            "text_vector": text_vector.detach().cpu().numpy(),
            "metadata_vector": metadata_vector.detach().cpu().numpy(),
            "combined_vector": combined_vector.detach().cpu().numpy()
        }
    
    def process_dataset(self, input_parquet, output_parquet):
        """
        Process an entire dataset, computing all three vectors for each item.
        """
        # Read input dataset
        df = wr.s3.read_parquet(path=input_parquet)
        
        # Process each item
        image_vectors = []
        text_vectors = []
        metadata_vectors = []
        combined_vectors = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processing item {idx}/{len(df)}")
            
            item = {
                "image_tensor": torch.tensor(row["image_embedding"]) if "image_embedding" in row else None,
                "image_path": row.get("image_path"),
                "image_url": row.get("image_url"),
                "description": row.get("description"),
                "metadata": row.get("metadata", {})
            }
            
            vectors = self.process_item(item)
            
            image_vectors.append(vectors["image_vector"])
            text_vectors.append(vectors["text_vector"])
            metadata_vectors.append(vectors["metadata_vector"])
            combined_vectors.append(vectors["combined_vector"])
        
        # Add vectors to dataframe
        df["image_vector"] = image_vectors
        df["text_vector"] = text_vectors
        df["metadata_vector"] = metadata_vectors
        df["combined_vector"] = combined_vectors
        
        # Save processed dataset
        wr.s3.to_parquet(df=df, path=output_parquet)
        
        print(f"Processed dataset saved to {output_parquet}")
    
    def update_qdrant(self, input_parquet):
        """
        Update Qdrant collection with the processed vectors.
        """
        # Read processed dataset
        df = wr.s3.read_parquet(path=input_parquet)
        
        # Prepare points for Qdrant
        points = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Preparing point {idx}/{len(df)} for Qdrant")
            
            # Use the combined vector as the primary vector
            point = {
                "id": row["id"] if "id" in row else idx,
                "vector": row["combined_vector"],
                "payload": {
                    "description": row.get("description", ""),
                    "metadata": row.get("metadata", {}),
                    # Store the individual vectors as well
                    "image_vector": row["image_vector"].tolist(),
                    "text_vector": row["text_vector"].tolist(),
                    "metadata_vector": row["metadata_vector"].tolist()
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
    
    def search(self, query, k=10, search_type="combined"):
        """
        Search the Qdrant collection using different search types.
        
        search_type can be "combined", "image", "text", or "metadata".
        """
        # Process query based on search type
        if search_type == "combined":
            # Create a dummy item with the query as description
            query_item = {"description": query, "metadata": {}}
            vectors = self.process_item(query_item)
            query_vector = vectors["combined_vector"]
        
        elif search_type == "text":
            # Get text embedding for query
            query_vector = self.text_embedder.get_embedding(query).detach().cpu().numpy()
        
        elif search_type == "metadata":
            # Parse query as metadata
            try:
                metadata = json.loads(query)
            except:
                metadata = {"query": query}
            
            query_vector = self.get_metadata_vector(metadata).detach().cpu().numpy()
        
        else:
            raise ValueError(f"Invalid search_type: {search_type}")
        
        # Search Qdrant
        results = self.qdrant_client.search(
            collection_name=self.qdrant_collection,
            query_vector=query_vector,
            limit=k
        )
        
        return results
    
    def train_weights(self, train_data, val_data, epochs=10, lr=0.01):
        """
        Train the weights of the vector combiner using training data.
        
        train_data and val_data should be lists of dictionaries with
        image_vector, text_vector, metadata_vector, and label.
        """
        # Only train if weights are learnable
        if not self.vector_combiner.learn_weights:
            print("Vector combiner has fixed weights, skipping training")
            return
        
        # Convert data to tensors
        train_image_vectors = torch.stack([torch.tensor(item["image_vector"]) for item in train_data])
        train_text_vectors = torch.stack([torch.tensor(item["text_vector"]) for item in train_data])
        train_metadata_vectors = torch.stack([torch.tensor(item["metadata_vector"]) for item in train_data])
        train_labels = torch.tensor([item["label"] for item in train_data])
        
        val_image_vectors = torch.stack([torch.tensor(item["image_vector"]) for item in val_data])
        val_text_vectors = torch.stack([torch.tensor(item["text_vector"]) for item in val_data])
        val_metadata_vectors = torch.stack([torch.tensor(item["metadata_vector"]) for item in val_data])
        val_labels = torch.tensor([item["label"] for item in val_data])
        
        # Define optimizer
        optimizer = torch.optim.Adam(self.vector_combiner.parameters(), lr=lr)
        
        # Define loss function (e.g., cosine similarity loss)
        loss_fn = torch.nn.CosineEmbeddingLoss()
        
        # Training loop
        for epoch in range(epochs):
            # Train
            self.vector_combiner.train()
            optimizer.zero_grad()
            
            # Forward pass
            train_combined_vectors = self.vector_combiner(
                train_image_vectors, train_text_vectors, train_metadata_vectors
            )
            
            # Compute loss (this is an example, adapt based on your task)
            # In a real scenario, you might have positive and negative pairs
            train_loss = loss_fn(
                train_combined_vectors,
                train_labels.unsqueeze(1).repeat(1, train_combined_vectors.shape[1]),
                torch.ones(train_combined_vectors.shape[0])
            )
            
            # Backward pass and optimize
            train_loss.backward()
            optimizer.step()
            
            # Validate
            self.vector_combiner.eval()
            with torch.no_grad():
                val_combined_vectors = self.vector_combiner(
                    val_image_vectors, val_text_vectors, val_metadata_vectors
                )
                
                val_loss = loss_fn(
                    val_combined_vectors,
                    val_labels.unsqueeze(1).repeat(1, val_combined_vectors.shape[1]),
                    torch.ones(val_combined_vectors.shape[0])
                )
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            print(f"Current weights: {self.vector_combiner.get_weights()}")

def main():
    parser = argparse.ArgumentParser(description="Implement three-vector architecture for image search")
    
    parser.add_argument("--input-parquet", type=str, required=True,
                        help="Path to input parquet file")
    parser.add_argument("--output-parquet", type=str, required=True,
                        help="Path to save processed data")
    parser.add_argument("--update-qdrant", action="store_true",
                        help="Update Qdrant collection with processed vectors")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost",
                        help="URL of Qdrant service")
    parser.add_argument("--qdrant-collection", type=str, default="images",
                        help="Name of Qdrant collection")
    parser.add_argument("--qdrant-api-key", type=str, default=None,
                        help="API key for Qdrant service")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP model to use for image embedding")
    parser.add_argument("--text-model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Text model to use for text and metadata embedding")
    parser.add_argument("--openai-api-key", type=str, default=None,
                        help="OpenAI API key for caption generation")
    parser.add_argument("--fixed-weights", action="store_true",
                        help="Use fixed weights for vector combination")
    parser.add_argument("--train-weights", action="store_true",
                        help="Train the weights for vector combination")
    parser.add_argument("--train-data", type=str, default=None,
                        help="Path to training data for weight training")
    parser.add_argument("--val-data", type=str, default=None,
                        help="Path to validation data for weight training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for weight training")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="Learning rate for weight training")
    
    args = parser.parse_args()
    
    # Initialize ThreeVectorEmbedding
    embedder = ThreeVectorEmbedding(
        clip_model_name=args.clip_model,
        text_model_name=args.text_model,
        openai_api_key=args.openai_api_key,
        fixed_weights=args.fixed_weights,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
        qdrant_api_key=args.qdrant_api_key
    )
    
    # Process dataset
    embedder.process_dataset(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet
    )
    
    # Update Qdrant if requested
    if args.update_qdrant:
        embedder.update_qdrant(input_parquet=args.output_parquet)
    
    # Train weights if requested
    if args.train_weights:
        if args.train_data is None or args.val_data is None:
            print("Warning: train_data or val_data not provided, skipping weight training")
        else:
            # Load training and validation data
            train_data = wr.s3.read_parquet(path=args.train_data).to_dict('records')
            val_data = wr.s3.read_parquet(path=args.val_data).to_dict('records')
            
            # Train weights
            embedder.train_weights(
                train_data=train_data,
                val_data=val_data,
                epochs=args.epochs,
                lr=args.learning_rate
            )
    
    # Print final weights
    print(f"Final weights: {embedder.vector_combiner.get_weights()}")

if __name__ == "__main__":
    main() 