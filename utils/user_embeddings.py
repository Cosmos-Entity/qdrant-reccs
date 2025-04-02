import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from qdrant_client import QdrantClient
import logging
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# Setup logging
logger = logging.getLogger(__name__)

class UserEmbeddingModel(nn.Module):
    """
    Neural network for generating user embeddings in the same latent space as content.
    """
    def __init__(self, vector_dim=1024, hidden_dim=512, dropout=0.2):
        super(UserEmbeddingModel, self).__init__()
        self.vector_dim = vector_dim
        
        # Define model layers
        self.transform = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vector_dim),
            nn.LayerNorm(vector_dim)
        )
    
    def forward(self, x, weights=None):
        """
        Generate user embedding from content vectors.
        
        Args:
            x: Content vectors [batch_size, vector_dim] or [n_items, vector_dim]
            weights: Optional weights for items
            
        Returns:
            User embedding
        """
        if x.shape[0] == 0:
            # No interactions, return zero vector
            return torch.zeros(self.vector_dim, device=x.device)
        
        if len(x.shape) == 3:
            # Handle batched input of multiple users with multiple items each
            batch_size, n_items, vector_dim = x.shape
            
            if weights is not None:
                # Apply weights to each user's items
                # weights shape: [batch_size, n_items]
                weights = weights.unsqueeze(-1)  # [batch_size, n_items, 1]
                weighted_x = x * weights
                sum_weights = weights.sum(dim=1, keepdim=True)
                aggregated = weighted_x.sum(dim=1) / (sum_weights + 1e-10)
            else:
                # Simple average for each user's items
                mask = (x.sum(dim=-1) != 0).float().unsqueeze(-1)  # [batch_size, n_items, 1]
                aggregated = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-10)
        
        else:
            # Single user with multiple items
            if weights is not None:
                # Apply weights
                weights = weights.unsqueeze(-1)  # [n_items, 1]
                weighted_x = x * weights
                aggregated = weighted_x.sum(dim=0) / (weights.sum() + 1e-10)
            else:
                # Simple average
                aggregated = x.mean(dim=0)
        
        # Transform aggregated vector to user embedding
        return self.transform(aggregated)

class UserEmbeddingSystem:
    """
    System for managing user embeddings in the same latent space as content.
    """
    def __init__(
        self, 
        vector_dim=1024,
        qdrant_url="http://localhost",
        qdrant_collection="elements",
        user_collection="users",
        qdrant_api_key=None,
        model_path=None,
        device=None
    ):
        # Vector dimension
        self.vector_dim = vector_dim
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = UserEmbeddingModel(vector_dim=vector_dim).to(self.device)
        
        # Load model if path provided
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded user embedding model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.content_collection = qdrant_collection
        self.user_collection = user_collection
        
        # Ensure user collection exists
        self._ensure_user_collection_exists()
    
    def _ensure_user_collection_exists(self):
        """Make sure the user collection exists in Qdrant."""
        try:
            # Check if collection exists
            self.qdrant_client.get_collection(self.user_collection)
            logger.info(f"Using existing collection: {self.user_collection}")
        except Exception:
            # Create collection if it doesn't exist
            self.qdrant_client.create_collection(
                collection_name=self.user_collection,
                vectors_config={"size": self.vector_dim, "distance": "Cosine"}
            )
            logger.info(f"Created new collection: {self.user_collection}")
    
    def get_user_embedding(self, user_id):
        """
        Get the current embedding for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            User embedding vector
        """
        try:
            # Try to retrieve from Qdrant
            response = self.qdrant_client.retrieve(
                collection_name=self.user_collection,
                ids=[user_id],
                with_vectors=True
            )
            
            if response and len(response) > 0:
                # User exists in database
                vector = np.array(response[0].vector)
                return vector
            else:
                # User not found
                logger.info(f"User {user_id} not found in database")
                return np.zeros(self.vector_dim)
                
        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {e}")
            return np.zeros(self.vector_dim)
    
    def update_user_embedding(self, user_id, element_ids=None, interactions=None, save=True):
        """
        Update a user's embedding based on their item interactions.
        
        Args:
            user_id: User ID
            element_ids: List of element IDs the user has interacted with
            interactions: Optional detailed interaction data
            save: Whether to save the embedding to Qdrant
            
        Returns:
            Updated user embedding
        """
        # Handle case where only element_ids are provided
        if element_ids and not interactions:
            interactions = [{'content_id': eid, 'weight': 1.0} for eid in element_ids]
        
        if not interactions:
            logger.warning(f"No interactions provided for user {user_id}")
            return self.get_user_embedding(user_id)
        
        # Get content vectors for items
        content_ids = [interaction['content_id'] for interaction in interactions]
        content_vectors = self._get_content_vectors(content_ids)
        
        if not content_vectors:
            logger.warning(f"No content vectors found for user {user_id}'s interactions")
            return self.get_user_embedding(user_id)
        
        # Prepare interaction data
        item_embeddings = []
        item_weights = []
        
        for interaction in interactions:
            content_id = interaction['content_id']
            if content_id in content_vectors:
                # Add vector
                item_embeddings.append(content_vectors[content_id])
                
                # Add weight if available
                weight = interaction.get('weight', 1.0)
                item_weights.append(weight)
        
        if not item_embeddings:
            logger.warning(f"No valid content embeddings for user {user_id}")
            return self.get_user_embedding(user_id)
        
        # Convert to tensors
        item_embeddings_tensor = torch.tensor(np.array(item_embeddings), dtype=torch.float32).to(self.device)
        
        # Convert weights if available
        if item_weights:
            item_weights_tensor = torch.tensor(np.array(item_weights), dtype=torch.float32).to(self.device)
        else:
            item_weights_tensor = None
        
        # Generate user embedding
        with torch.no_grad():
            user_embedding = self.model(item_embeddings_tensor, item_weights_tensor)
            user_embedding_np = user_embedding.cpu().numpy()
        
        # Save to database if requested
        if save:
            self._save_user_embedding(user_id, user_embedding_np, interactions)
        
        return user_embedding_np
    
    def _get_content_vectors(self, content_ids):
        """
        Retrieve content vectors from Qdrant.
        
        Args:
            content_ids: List of content IDs
            
        Returns:
            Dictionary mapping content IDs to vectors
        """
        try:
            # Batch retrieval by chunks of 100
            all_vectors = {}
            batch_size = 100
            
            for i in range(0, len(content_ids), batch_size):
                batch_ids = content_ids[i:i+batch_size]
                
                response = self.qdrant_client.retrieve(
                    collection_name=self.content_collection,
                    ids=batch_ids,
                    with_vectors=True
                )
                
                for point in response:
                    all_vectors[point.id] = np.array(point.vector)
            
            return all_vectors
            
        except Exception as e:
            logger.error(f"Error retrieving content vectors: {e}")
            return {}
    
    def _save_user_embedding(self, user_id, embedding, interactions=None):
        """
        Save user embedding to Qdrant.
        
        Args:
            user_id: User ID
            embedding: User embedding vector
            interactions: Optional interaction data for payload
        """
        try:
            # Prepare payload
            payload = {}
            
            if interactions:
                # Store last update timestamp
                payload['last_updated'] = datetime.now().isoformat()
                
                # Store interaction summary
                interaction_counts = defaultdict(int)
                for interaction in interactions:
                    interaction_type = interaction.get('type', 'view')
                    interaction_counts[interaction_type] += 1
                
                payload['interaction_summary'] = dict(interaction_counts)
                
                # Store number of interactions
                payload['num_interactions'] = len(interactions)
            
            # Upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.user_collection,
                points=[{
                    'id': user_id,
                    'vector': embedding.tolist(),
                    'payload': payload
                }]
            )
            
            logger.info(f"Saved embedding for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error saving user embedding: {e}")
    
    def process_user_data(self, user_data, batch_size=100):
        """
        Process user data from user snapshot.
        
        Args:
            user_data: Dictionary mapping user_ids to element_ids
            batch_size: Batch size for processing
            
        Returns:
            Number of users processed
        """
        logger.info(f"Processing embeddings for {len(user_data)} users")
        user_count = 0
        
        # Process in batches
        user_ids = list(user_data.keys())
        for i in range(0, len(user_ids), batch_size):
            batch_user_ids = user_ids[i:i+batch_size]
            
            for user_id in tqdm(batch_user_ids, desc=f"Batch {i//batch_size + 1}/{len(user_ids)//batch_size + 1}"):
                element_ids = user_data[user_id]
                self.update_user_embedding(user_id, element_ids=element_ids)
                user_count += 1
        
        logger.info(f"Processed embeddings for {user_count} users")
        return user_count
    
    def get_similar_users(self, user_id, k=10):
        """
        Find users similar to the given user.
        
        Args:
            user_id: User ID
            k: Number of similar users to return
            
        Returns:
            List of similar user IDs with scores
        """
        try:
            # Get user embedding
            user_vector = self.get_user_embedding(user_id)
            
            if np.all(user_vector == 0):
                logger.warning(f"No embedding found for user {user_id}")
                return []
            
            # Search for similar users
            results = self.qdrant_client.search(
                collection_name=self.user_collection,
                query_vector=user_vector.tolist(),
                limit=k+1  # +1 because the user will match themselves
            )
            
            # Filter out the query user
            similar_users = []
            for res in results:
                if res.id != user_id:
                    similar_users.append({
                        'user_id': res.id,
                        'score': res.score
                    })
            
            return similar_users[:k]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return []
    
    def get_recommendations(self, user_id, k=10, exclude_ids=None):
        """
        Get content recommendations for a user.
        
        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_ids: Content IDs to exclude (e.g., already seen items)
            
        Returns:
            List of recommended content IDs with scores
        """
        try:
            # Get user embedding
            user_vector = self.get_user_embedding(user_id)
            
            if np.all(user_vector == 0):
                logger.warning(f"No embedding found for user {user_id}")
                return []
            
            # Prepare filter
            filter_condition = None
            if exclude_ids:
                filter_condition = {
                    "must_not": [
                        {"id": {"in": exclude_ids}}
                    ]
                }
            
            # Search for similar content
            results = self.qdrant_client.search(
                collection_name=self.content_collection,
                query_vector=user_vector.tolist(),
                limit=k,
                filter=filter_condition
            )
            
            # Format results
            recommendations = []
            for res in results:
                recommendations.append({
                    'content_id': res.id,
                    'score': res.score
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def save_model(self, path):
        """Save the user embedding model to disk."""
        try:
            # Create directory if needed
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            torch.save(self.model.state_dict(), path)
            logger.info(f"Saved model to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")

# Process user snapshot data
def process_user_snapshot_data(snapshot_dir):
    """
    Process user data from the extracted snapshot.
    
    Args:
        snapshot_dir: Path to the extracted snapshot directory
    
    Returns:
        Dictionary mapping user_id to element_ids
    """
    import os
    import tarfile
    import json
    
    user_data = {}
    segments_dir = os.path.join(snapshot_dir, "0/segments")
    
    # Check if directory exists
    if not os.path.exists(segments_dir):
        logger.error(f"Segments directory not found: {segments_dir}")
        return user_data
    
    # Process each segment file
    for filename in os.listdir(segments_dir):
        if filename.endswith('.tar'):
            segment_path = os.path.join(segments_dir, filename)
            
            try:
                with tarfile.open(segment_path, 'r') as tar:
                    # Find payload files
                    payload_files = [m for m in tar.getmembers() if 'payload_storage/page_0.dat' in m.name]
                    for payload_file in payload_files:
                        f = tar.extractfile(payload_file)
                        if f:
                            data = f.read()
                            
                            # Skip header (first 16 bytes)
                            payload_data = data[16:]
                            
                            # Extract JSON data
                            start = 0
                            while start < len(payload_data):
                                try:
                                    json_start = payload_data.index(b'{', start)
                                    # Find matching closing brace
                                    brace_count = 1
                                    pos = json_start + 1
                                    while brace_count > 0 and pos < len(payload_data):
                                        if payload_data[pos] == ord('{'):
                                            brace_count += 1
                                        elif payload_data[pos] == ord('}'):
                                            brace_count -= 1
                                        pos += 1
                                    
                                    if brace_count == 0:
                                        json_data = payload_data[json_start:pos]
                                        try:
                                            # Parse JSON
                                            user = json.loads(json_data.decode('utf-8'))
                                            if 'user_id' in user and 'element_ids' in user:
                                                user_data[user['user_id']] = user['element_ids']
                                        except:
                                            pass
                                        start = pos
                                    else:
                                        start = json_start + 1
                                except ValueError:
                                    break
            except Exception as e:
                logger.error(f"Error processing segment {filename}: {e}")
    
    logger.info(f"Processed {len(user_data)} users from snapshot")
    return user_data

# Example usage
if __name__ == "__main__":
    # Initialize system
    system = UserEmbeddingSystem()
    
    # Test with some sample data
    user_id = "test_user_1"
    element_ids = ["item1", "item2", "item3"]
    
    # Generate and save embedding
    embedding = system.update_user_embedding(user_id, element_ids=element_ids)
    
    print(f"Generated embedding for user {user_id} with shape {embedding.shape}")
    
    # Get recommendations
    recommendations = system.get_recommendations(user_id, k=5)
    print(f"Top recommendations for user {user_id}:")
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. Item {rec['content_id']} (score: {rec['score']:.4f})") 