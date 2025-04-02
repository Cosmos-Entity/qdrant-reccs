import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Any
from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logger = logging.getLogger(__name__)

class ThreeVectorEmbedding:
    """
    Implementation of the three-vector architecture for embeddings:
    1. Image Vector: CLIP embedding of image
    2. Text Description Vector: Generated via LLM caption or description
    3. Metadata Vector: Embedding of structured metadata
    """
    def __init__(
        self,
        qdrant_url: str = "http://localhost",
        qdrant_collection: str = "images",
        qdrant_api_key: Optional[str] = None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        text_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        image_weight: float = 0.5,
        text_weight: float = 0.3,
        metadata_weight: float = 0.2,
        device: Optional[str] = None
    ):
        """
        Initialize the three-vector embedding system.
        
        Args:
            qdrant_url: URL of Qdrant service
            qdrant_collection: Name of Qdrant collection
            qdrant_api_key: Optional API key for Qdrant
            clip_model_name: Name of CLIP model to use
            text_model_name: Name of text embedding model
            image_weight: Weight for image vector in combined representation
            text_weight: Weight for text vector in combined representation
            metadata_weight: Weight for metadata vector in combined representation
            device: Device to run models on (defaults to CUDA if available)
        """
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.qdrant_collection = qdrant_collection
        
        # Set weights and normalize
        total_weight = image_weight + text_weight + metadata_weight
        self.image_weight = image_weight / total_weight
        self.text_weight = text_weight / total_weight
        self.metadata_weight = metadata_weight / total_weight
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._init_models(clip_model_name, text_model_name)
        
        # Vector dimensions
        self.image_dim = 1024  # Default CLIP image dim
        self.text_dim = 768    # Default text embedding dim
        self.metadata_dim = 768  # Metadata embedding dim
        
        # Neural combiner
        self._init_combiner()
    
    def _init_models(self, clip_model_name, text_model_name):
        """Initialize the embedding models."""
        try:
            # Load CLIP model for image vectors
            logger.info(f"Loading CLIP model: {clip_model_name}")
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            
            # Load text embedding model
            logger.info(f"Loading text model: {text_model_name}")
            self.text_model = SentenceTransformer(text_model_name, device=self.device)
            
            # Get actual dimensions from models
            with torch.no_grad():
                # Test input to get dimensions
                dummy_img = torch.zeros(1, 3, 224, 224).to(self.device)
                dummy_text = ["test"]
                
                # Get image dim from CLIP
                img_features = self.clip_model.get_image_features(pixel_values=dummy_img)
                self.image_dim = img_features.shape[-1]
                
                # Get text dim
                text_features = self.text_model.encode(dummy_text)
                self.text_dim = text_features.shape[-1]
                
            logger.info(f"Model dimensions - Image: {self.image_dim}, Text: {self.text_dim}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Use default dimensions if model loading fails
            pass
    
    def _init_combiner(self):
        """Initialize the neural network for combining vectors."""
        # Image vector processor
        self.image_processor = nn.Sequential(
            nn.Linear(self.image_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # Text vector processor
        self.text_processor = nn.Sequential(
            nn.Linear(self.text_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # Metadata vector processor
        self.metadata_processor = nn.Sequential(
            nn.Linear(self.metadata_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # Vector combiner
        self.combiner = nn.Sequential(
            nn.Linear(512 * 3, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )
        
        # Move to device
        self.image_processor.to(self.device)
        self.text_processor.to(self.device)
        self.metadata_processor.to(self.device)
        self.combiner.to(self.device)
    
    def get_image_vector(self, image=None, image_id=None):
        """
        Get image vector using CLIP.
        
        Args:
            image: PIL Image or tensor
            image_id: ID of image in Qdrant
        
        Returns:
            Image embedding vector
        """
        if image is not None:
            # Process image with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            return image_features.cpu().numpy().squeeze()
        
        elif image_id is not None:
            # Retrieve from Qdrant
            try:
                response = self.qdrant_client.retrieve(
                    collection_name=self.qdrant_collection,
                    ids=[image_id],
                    with_payload=False,
                    with_vectors=True
                )
                
                if response and len(response) > 0:
                    return np.array(response[0].vector)
                
                logger.warning(f"Image {image_id} not found in Qdrant")
                return np.zeros(self.image_dim)
                
            except Exception as e:
                logger.error(f"Error retrieving image vector: {e}")
                return np.zeros(self.image_dim)
        
        else:
            logger.warning("No image or image_id provided")
            return np.zeros(self.image_dim)
    
    def get_text_vector(self, text=None, item_id=None):
        """
        Get text vector using text embedding model.
        
        Args:
            text: Text description
            item_id: ID of item in Qdrant
        
        Returns:
            Text embedding vector
        """
        if text is not None:
            # Encode text
            return self.text_model.encode(text)
        
        elif item_id is not None:
            # Retrieve from Qdrant
            try:
                response = self.qdrant_client.retrieve(
                    collection_name=self.qdrant_collection,
                    ids=[item_id],
                    with_payload=True
                )
                
                if response and len(response) > 0:
                    # Check if payload has description
                    payload = response[0].payload
                    if payload and "description" in payload:
                        return self.text_model.encode(payload["description"])
                
                logger.warning(f"Item {item_id} not found or has no description")
                return np.zeros(self.text_dim)
                
            except Exception as e:
                logger.error(f"Error retrieving text description: {e}")
                return np.zeros(self.text_dim)
        
        else:
            logger.warning("No text or item_id provided")
            return np.zeros(self.text_dim)
    
    def get_metadata_vector(self, metadata=None, item_id=None):
        """
        Get metadata vector.
        
        Args:
            metadata: Metadata dictionary
            item_id: ID of item in Qdrant
        
        Returns:
            Metadata embedding vector
        """
        if metadata is not None:
            # Convert metadata to string
            metadata_str = self._metadata_to_string(metadata)
            # Encode metadata string
            return self.text_model.encode(metadata_str)
        
        elif item_id is not None:
            # Retrieve from Qdrant
            try:
                response = self.qdrant_client.retrieve(
                    collection_name=self.qdrant_collection,
                    ids=[item_id],
                    with_payload=True
                )
                
                if response and len(response) > 0:
                    # Check if payload has metadata
                    payload = response[0].payload
                    if payload and "metadata" in payload:
                        metadata_str = self._metadata_to_string(payload["metadata"])
                        return self.text_model.encode(metadata_str)
                
                logger.warning(f"Item {item_id} not found or has no metadata")
                return np.zeros(self.metadata_dim)
                
            except Exception as e:
                logger.error(f"Error retrieving metadata: {e}")
                return np.zeros(self.metadata_dim)
        
        else:
            logger.warning("No metadata or item_id provided")
            return np.zeros(self.metadata_dim)
    
    def _metadata_to_string(self, metadata):
        """Convert metadata dictionary to string representation."""
        if not metadata:
            return ""
        
        metadata_parts = []
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                value_str = ", ".join(str(v) for v in value)
                metadata_parts.append(f"{key}: {value_str}")
            elif isinstance(value, dict):
                # Recursive handling of nested dictionaries
                nested_str = "; ".join(f"{k}: {v}" for k, v in value.items())
                metadata_parts.append(f"{key}: {{{nested_str}}}")
            else:
                metadata_parts.append(f"{key}: {value}")
        
        return ". ".join(metadata_parts)
    
    def get_combined_vector(self, image_vector, text_vector, metadata_vector, use_neural=True):
        """
        Combine three vectors into a single representation.
        
        Args:
            image_vector: Image embedding vector
            text_vector: Text embedding vector
            metadata_vector: Metadata embedding vector
            use_neural: Whether to use neural combiner (if False, uses weighted average)
        
        Returns:
            Combined vector
        """
        # Ensure vectors are numpy arrays with correct dimensions
        if len(image_vector.shape) == 1:
            image_vector = image_vector.reshape(1, -1)
        if len(text_vector.shape) == 1:
            text_vector = text_vector.reshape(1, -1)
        if len(metadata_vector.shape) == 1:
            metadata_vector = metadata_vector.reshape(1, -1)
        
        if use_neural:
            # Convert to tensors
            image_tensor = torch.tensor(image_vector, dtype=torch.float32).to(self.device)
            text_tensor = torch.tensor(text_vector, dtype=torch.float32).to(self.device)
            metadata_tensor = torch.tensor(metadata_vector, dtype=torch.float32).to(self.device)
            
            # Apply neural processing
            with torch.no_grad():
                # Process each vector
                processed_image = self.image_processor(image_tensor)
                processed_text = self.text_processor(text_tensor)
                processed_metadata = self.metadata_processor(metadata_tensor)
                
                # Concatenate processed vectors
                combined = torch.cat([processed_image, processed_text, processed_metadata], dim=1)
                
                # Apply final combination
                result = self.combiner(combined)
                
                # Return numpy array
                return result.cpu().numpy().squeeze()
        else:
            # Simple weighted average
            combined = (
                self.image_weight * image_vector +
                self.text_weight * text_vector +
                self.metadata_weight * metadata_vector
            )
            
            # Normalize
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
                
            return combined.squeeze()
    
    def get_item_vectors(self, item_id):
        """
        Get all three vectors for an item.
        
        Args:
            item_id: ID of item in Qdrant
        
        Returns:
            Dictionary with image_vector, text_vector, metadata_vector, and combined_vector
        """
        # Get individual vectors
        image_vector = self.get_image_vector(image_id=item_id)
        text_vector = self.get_text_vector(item_id=item_id)
        metadata_vector = self.get_metadata_vector(item_id=item_id)
        
        # Combine vectors
        combined_vector = self.get_combined_vector(image_vector, text_vector, metadata_vector)
        
        return {
            "image_vector": image_vector,
            "text_vector": text_vector,
            "metadata_vector": metadata_vector,
            "combined_vector": combined_vector
        }
    
    def save_item_vectors(self, item_id, image_vector=None, text_vector=None, metadata_vector=None, combined_vector=None):
        """
        Save all vectors for an item to Qdrant.
        
        Args:
            item_id: ID of item
            image_vector: Image embedding vector (optional)
            text_vector: Text embedding vector (optional)
            metadata_vector: Metadata embedding vector (optional)
            combined_vector: Combined vector (optional)
        
        Returns:
            Success status
        """
        try:
            # Get existing item if available
            existing_item = None
            try:
                response = self.qdrant_client.retrieve(
                    collection_name=self.qdrant_collection,
                    ids=[item_id],
                    with_payload=True,
                    with_vectors=True
                )
                if response and len(response) > 0:
                    existing_item = response[0]
            except:
                pass
            
            # Prepare payload
            payload = {}
            if existing_item and hasattr(existing_item, 'payload'):
                payload = existing_item.payload or {}
            
            # Add vectors to payload
            vectors_payload = payload.get("vectors", {})
            
            if image_vector is not None:
                vectors_payload["image_vector"] = image_vector.tolist() if isinstance(image_vector, np.ndarray) else image_vector
            
            if text_vector is not None:
                vectors_payload["text_vector"] = text_vector.tolist() if isinstance(text_vector, np.ndarray) else text_vector
            
            if metadata_vector is not None:
                vectors_payload["metadata_vector"] = metadata_vector.tolist() if isinstance(metadata_vector, np.ndarray) else metadata_vector
            
            payload["vectors"] = vectors_payload
            
            # Use combined vector as primary vector, or compute it if not provided
            if combined_vector is None and image_vector is not None and text_vector is not None and metadata_vector is not None:
                combined_vector = self.get_combined_vector(image_vector, text_vector, metadata_vector)
            
            if combined_vector is not None:
                # Ensure it's a list
                vector = combined_vector.tolist() if isinstance(combined_vector, np.ndarray) else combined_vector
                
                # Upsert to Qdrant
                self.qdrant_client.upsert(
                    collection_name=self.qdrant_collection,
                    points=[{
                        "id": item_id,
                        "vector": vector,
                        "payload": payload
                    }]
                )
                
                return True
            else:
                logger.warning(f"No combined vector for item {item_id}")
                return False
        
        except Exception as e:
            logger.error(f"Error saving item vectors: {e}")
            return False
    
    def process_batch(self, item_ids, save=True):
        """
        Process a batch of items to generate and optionally save all three vectors.
        
        Args:
            item_ids: List of item IDs
            save: Whether to save results to Qdrant
        
        Returns:
            Dictionary mapping item IDs to their vectors
        """
        results = {}
        for item_id in item_ids:
            vectors = self.get_item_vectors(item_id)
            results[item_id] = vectors
            
            if save:
                self.save_item_vectors(
                    item_id=item_id,
                    image_vector=vectors["image_vector"],
                    text_vector=vectors["text_vector"],
                    metadata_vector=vectors["metadata_vector"],
                    combined_vector=vectors["combined_vector"]
                )
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize three-vector embedding
    embedder = ThreeVectorEmbedding()
    
    # Test with a sample item
    item_id = "test_item_1"
    
    # Sample vectors (random for testing)
    image_vector = np.random.rand(1024)
    text_vector = np.random.rand(768)
    metadata_vector = np.random.rand(768)
    
    # Get combined vector
    combined = embedder.get_combined_vector(image_vector, text_vector, metadata_vector)
    
    print(f"Combined vector shape: {combined.shape}")
    print(f"Combined vector norm: {np.linalg.norm(combined)}") 