import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import awswrangler as wr
from qdrant_client import QdrantClient
from typing import List, Dict, Tuple, Optional, Union
import json
import os
from datetime import datetime
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class UserEmbeddingModel(nn.Module):
    """
    Neural network for generating user embeddings in the same latent space as content.
    """
    def __init__(self, vector_dim=1024, hidden_dim=512):
        super(UserEmbeddingModel, self).__init__()
        self.vector_dim = vector_dim
        
        # Transform interaction embeddings into user embeddings
        self.transform = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, vector_dim),
            nn.LayerNorm(vector_dim)
        )
    
    def forward(self, interaction_embeddings, interaction_weights=None):
        """
        Generate user embedding from content interaction embeddings.
        
        Args:
            interaction_embeddings: Tensor of shape [n_interactions, vector_dim]
            interaction_weights: Optional tensor of shape [n_interactions] for weighted aggregation
        
        Returns:
            User embedding tensor of shape [vector_dim]
        """
        if interaction_embeddings.shape[0] == 0:
            # No interactions, return zero vector
            return torch.zeros(self.vector_dim)
        
        if interaction_weights is not None:
            # Weighted average of interaction embeddings
            weights = interaction_weights.unsqueeze(1)
            weighted_embeddings = interaction_embeddings * weights
            aggregated = weighted_embeddings.sum(dim=0) / weights.sum()
        else:
            # Simple average of interaction embeddings
            aggregated = interaction_embeddings.mean(dim=0)
        
        # Transform aggregated embeddings into user embedding
        user_embedding = self.transform(aggregated)
        
        return user_embedding

class InteractionWeightModel(nn.Module):
    """
    Model for determining the weight of each interaction based on various factors.
    """
    def __init__(self, n_interaction_types=5, n_time_buckets=10):
        super(InteractionWeightModel, self).__init__()
        
        # Embeddings for different interaction types (view, like, share, comment, save)
        self.interaction_type_embedding = nn.Embedding(n_interaction_types, 32)
        
        # Embeddings for time buckets (recency of interaction)
        self.time_embedding = nn.Embedding(n_time_buckets, 32)
        
        # MLP for computing weights
        self.weight_mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, interaction_types, interaction_times):
        """
        Compute weights for interactions.
        
        Args:
            interaction_types: Tensor of interaction type indices
            interaction_times: Tensor of interaction time bucket indices
        
        Returns:
            Tensor of interaction weights
        """
        type_emb = self.interaction_type_embedding(interaction_types)
        time_emb = self.time_embedding(interaction_times)
        
        # Concatenate embeddings
        combined = torch.cat([type_emb, time_emb], dim=1)
        
        # Compute weights
        weights = self.weight_mlp(combined).squeeze()
        
        return weights

class UserEmbeddingSystem:
    """
    System for managing user embeddings in the same latent space as content.
    """
    def __init__(
        self, 
        vector_dim=1024,
        qdrant_url="http://localhost",
        qdrant_collection="images",
        user_collection="users",
        qdrant_api_key=None,
        model_path=None
    ):
        self.vector_dim = vector_dim
        
        # Initialize models
        self.user_model = UserEmbeddingModel(vector_dim=vector_dim)
        self.weight_model = InteractionWeightModel()
        
        # Load models if path provided
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.user_model.load_state_dict(checkpoint['user_model'])
            self.weight_model.load_state_dict(checkpoint['weight_model'])
            print(f"Loaded models from {model_path}")
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.content_collection = qdrant_collection
        self.user_collection = user_collection
        
        # Ensure user collection exists
        self._ensure_user_collection_exists()
        
        # Interaction type mapping
        self.interaction_types = {
            'view': 0,
            'like': 1,
            'share': 2,
            'comment': 3,
            'save': 4
        }
    
    def _ensure_user_collection_exists(self):
        """
        Ensure the user collection exists in Qdrant.
        """
        try:
            self.qdrant_client.get_collection(self.user_collection)
        except Exception:
            # Collection doesn't exist, create it
            self.qdrant_client.create_collection(
                collection_name=self.user_collection,
                vectors_config={"size": self.vector_dim, "distance": "Cosine"}
            )
            print(f"Created new collection: {self.user_collection}")
    
    def _time_to_bucket(self, timestamp, n_buckets=10, max_age_days=365):
        """
        Convert timestamp to a time bucket index.
        
        Args:
            timestamp: Datetime timestamp of interaction
            n_buckets: Number of time buckets
            max_age_days: Maximum age to consider (older interactions go in oldest bucket)
        
        Returns:
            Bucket index (0 for most recent, n_buckets-1 for oldest)
        """
        # Calculate age in days
        now = datetime.now()
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                # If parsing fails, assume it's old
                return n_buckets - 1
        
        age_days = (now - timestamp).days
        
        # Cap at max age
        age_days = min(age_days, max_age_days)
        
        # Convert to bucket (0 is most recent)
        bucket = int(age_days / max_age_days * n_buckets)
        return min(bucket, n_buckets - 1)
    
    def get_user_embedding(self, user_id):
        """
        Get the current embedding for a user.
        
        Args:
            user_id: Unique identifier for the user
        
        Returns:
            User embedding vector (numpy array)
        """
        try:
            # Try to retrieve from Qdrant
            response = self.qdrant_client.retrieve(
                collection_name=self.user_collection,
                ids=[user_id]
            )
            
            if response and len(response) > 0:
                # User exists in the database
                return np.array(response[0].vector)
            else:
                # User not found, return zero vector
                return np.zeros(self.vector_dim)
        
        except Exception as e:
            print(f"Error retrieving user {user_id}: {e}")
            return np.zeros(self.vector_dim)
    
    def update_user_embedding(self, user_id, interactions, save=True):
        """
        Update user embedding based on content interactions.
        
        Args:
            user_id: Unique identifier for the user
            interactions: List of interaction dictionaries containing:
                - content_id: ID of the content item
                - type: Type of interaction (view, like, share, etc.)
                - timestamp: When the interaction occurred
            save: Whether to save the updated embedding to the database
        
        Returns:
            Updated user embedding vector (numpy array)
        """
        if not interactions:
            return self.get_user_embedding(user_id)
        
        # Get content embeddings for all interactions
        content_ids = [interaction['content_id'] for interaction in interactions]
        content_vectors = self._get_content_vectors(content_ids)
        
        if not content_vectors:
            return self.get_user_embedding(user_id)
        
        # Prepare interaction data
        interaction_embeddings = []
        interaction_types = []
        interaction_times = []
        
        for idx, interaction in enumerate(interactions):
            content_id = interaction['content_id']
            if content_id in content_vectors:
                # Add content vector
                interaction_embeddings.append(content_vectors[content_id])
                
                # Add interaction type
                int_type = self.interaction_types.get(
                    interaction['type'].lower(), 0  # Default to 'view'
                )
                interaction_types.append(int_type)
                
                # Add interaction time bucket
                time_bucket = self._time_to_bucket(interaction.get('timestamp', datetime.now()))
                interaction_times.append(time_bucket)
        
        if not interaction_embeddings:
            return self.get_user_embedding(user_id)
        
        # Convert to tensors
        interaction_embeddings = torch.tensor(np.array(interaction_embeddings), dtype=torch.float32)
        interaction_types = torch.tensor(interaction_types, dtype=torch.long)
        interaction_times = torch.tensor(interaction_times, dtype=torch.long)
        
        # Compute interaction weights
        with torch.no_grad():
            weights = self.weight_model(interaction_types, interaction_times)
        
        # Generate user embedding
        with torch.no_grad():
            user_embedding = self.user_model(interaction_embeddings, weights)
        
        # Convert to numpy
        user_embedding_np = user_embedding.detach().cpu().numpy()
        
        # Save to database if requested
        if save:
            self._save_user_embedding(user_id, user_embedding_np, interactions)
        
        return user_embedding_np
    
    def _get_content_vectors(self, content_ids):
        """
        Retrieve content vectors from Qdrant.
        
        Args:
            content_ids: List of content IDs to retrieve
        
        Returns:
            Dictionary mapping content IDs to vectors
        """
        try:
            response = self.qdrant_client.retrieve(
                collection_name=self.content_collection,
                ids=content_ids
            )
            
            content_vectors = {}
            for point in response:
                content_vectors[point.id] = np.array(point.vector)
            
            return content_vectors
        
        except Exception as e:
            print(f"Error retrieving content vectors: {e}")
            return {}
    
    def _save_user_embedding(self, user_id, embedding, interactions=None):
        """
        Save user embedding to Qdrant.
        
        Args:
            user_id: Unique identifier for the user
            embedding: User embedding vector
            interactions: Optional list of interactions that generated this embedding
        """
        payload = {}
        
        if interactions:
            # Store last update time
            payload['last_update'] = datetime.now().isoformat()
            
            # Store a summary of interactions
            interaction_summary = defaultdict(int)
            for interaction in interactions:
                int_type = interaction.get('type', 'view').lower()
                interaction_summary[int_type] += 1
            
            payload['interaction_summary'] = dict(interaction_summary)
        
        # Store user vector in Qdrant
        try:
            self.qdrant_client.upsert(
                collection_name=self.user_collection,
                points=[{
                    'id': user_id,
                    'vector': embedding.tolist(),
                    'payload': payload
                }]
            )
        except Exception as e:
            print(f"Error saving user embedding: {e}")
    
    def recommend_for_user(self, user_id, k=10, exclude_ids=None, filter_condition=None):
        """
        Generate content recommendations for a user.
        
        Args:
            user_id: Unique identifier for the user
            k: Number of recommendations to return
            exclude_ids: List of content IDs to exclude (e.g., already viewed)
            filter_condition: Optional filter condition for Qdrant search
        
        Returns:
            List of recommended content items
        """
        # Get user embedding
        user_vector = self.get_user_embedding(user_id)
        
        if np.all(user_vector == 0):
            # New user with no embedding, return popular items
            return self._get_popular_content(k)
        
        # Prepare search filter
        search_filter = None
        if exclude_ids:
            search_filter = {
                "must_not": [
                    {"id": {"in": exclude_ids}}
                ]
            }
        
        if filter_condition:
            if search_filter:
                search_filter["must"] = [filter_condition]
            else:
                search_filter = {"must": [filter_condition]}
        
        # Search for similar content
        try:
            results = self.qdrant_client.search(
                collection_name=self.content_collection,
                query_vector=user_vector.tolist(),
                limit=k,
                filter=search_filter
            )
            
            recommendations = []
            for res in results:
                recommendations.append({
                    'id': res.id,
                    'score': res.score,
                    'metadata': res.payload.get('metadata', {}),
                    'description': res.payload.get('description', '')
                })
            
            return recommendations
        
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []
    
    def _get_popular_content(self, k=10):
        """
        Get popular content for cold-start recommendations.
        
        In a real system, this would query a pre-computed list of popular items.
        Here we'll just return random items as a placeholder.
        
        Args:
            k: Number of items to return
        
        Returns:
            List of popular content items
        """
        try:
            # In a real system, you would have a more sophisticated approach
            # This is just a placeholder returning random items
            results = self.qdrant_client.scroll(
                collection_name=self.content_collection,
                limit=k,
                with_payload=True,
                with_vectors=False
            )[0]
            
            recommendations = []
            for res in results:
                recommendations.append({
                    'id': res.id,
                    'score': 1.0,  # Placeholder score
                    'metadata': res.payload.get('metadata', {}),
                    'description': res.payload.get('description', '')
                })
            
            return recommendations
        
        except Exception as e:
            print(f"Error getting popular content: {e}")
            return []
    
    def find_similar_users(self, user_id, k=10):
        """
        Find users similar to the given user.
        
        Args:
            user_id: Unique identifier for the reference user
            k: Number of similar users to return
        
        Returns:
            List of similar user IDs with similarity scores
        """
        # Get user embedding
        user_vector = self.get_user_embedding(user_id)
        
        if np.all(user_vector == 0):
            # User has no embedding
            return []
        
        # Search for similar users
        try:
            results = self.qdrant_client.search(
                collection_name=self.user_collection,
                query_vector=user_vector.tolist(),
                limit=k+1  # +1 because the user will match themselves
            )
            
            similar_users = []
            for res in results:
                # Skip the user themselves
                if res.id == user_id:
                    continue
                
                similar_users.append({
                    'user_id': res.id,
                    'similarity': res.score
                })
            
            return similar_users[:k]  # Ensure we return at most k users
        
        except Exception as e:
            print(f"Error finding similar users: {e}")
            return []
    
    def batch_update_users(self, user_interactions, batch_size=100):
        """
        Update embeddings for multiple users in batch.
        
        Args:
            user_interactions: Dictionary mapping user IDs to lists of interactions
            batch_size: Number of users to process in each batch
        """
        user_ids = list(user_interactions.keys())
        
        for i in range(0, len(user_ids), batch_size):
            batch_user_ids = user_ids[i:i+batch_size]
            
            for user_id in tqdm(batch_user_ids, desc=f"Processing batch {i//batch_size + 1}"):
                interactions = user_interactions[user_id]
                self.update_user_embedding(user_id, interactions)
    
    def save_models(self, path):
        """
        Save models to disk.
        
        Args:
            path: Path to save the models
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'user_model': self.user_model.state_dict(),
            'weight_model': self.weight_model.state_dict()
        }, path)
        print(f"Models saved to {path}")
    
    def train(self, train_data, val_data=None, epochs=10, batch_size=64, lr=0.001):
        """
        Train the user embedding and interaction weight models.
        
        Args:
            train_data: List of training examples, each with:
                - user_id: User identifier
                - interactions: List of interaction dictionaries
                - target_embedding: Target embedding for the user (e.g., future interactions)
            val_data: Optional validation data in the same format
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
        
        Returns:
            Dictionary of training metrics
        """
        # Move models to training mode
        self.user_model.train()
        self.weight_model.train()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            list(self.user_model.parameters()) + 
            list(self.weight_model.parameters()),
            lr=lr
        )
        
        # Loss function
        loss_fn = nn.MSELoss()
        
        # Training metrics
        metrics = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Process mini-batches
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                batch_loss = 0
                
                # Process each example in the batch
                for example in batch:
                    user_id = example['user_id']
                    interactions = example['interactions']
                    target_embedding = torch.tensor(example['target_embedding'], dtype=torch.float32)
                    
                    # Get content embeddings
                    content_ids = [interaction['content_id'] for interaction in interactions]
                    content_vectors = self._get_content_vectors(content_ids)
                    
                    if not content_vectors:
                        continue
                    
                    # Prepare interaction data
                    interaction_embeddings = []
                    interaction_types = []
                    interaction_times = []
                    
                    for interaction in interactions:
                        content_id = interaction['content_id']
                        if content_id in content_vectors:
                            interaction_embeddings.append(content_vectors[content_id])
                            
                            int_type = self.interaction_types.get(
                                interaction['type'].lower(), 0
                            )
                            interaction_types.append(int_type)
                            
                            time_bucket = self._time_to_bucket(interaction.get('timestamp', datetime.now()))
                            interaction_times.append(time_bucket)
                    
                    if not interaction_embeddings:
                        continue
                    
                    # Convert to tensors
                    interaction_embeddings = torch.tensor(np.array(interaction_embeddings), dtype=torch.float32)
                    interaction_types = torch.tensor(interaction_types, dtype=torch.long)
                    interaction_times = torch.tensor(interaction_times, dtype=torch.long)
                    
                    # Forward pass
                    weights = self.weight_model(interaction_types, interaction_times)
                    user_embedding = self.user_model(interaction_embeddings, weights)
                    
                    # Compute loss
                    loss = loss_fn(user_embedding, target_embedding)
                    batch_loss += loss.item()
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                if batch:
                    batch_loss /= len(batch)
                    epoch_loss += batch_loss
                    num_batches += 1
            
            # Compute average loss for the epoch
            if num_batches > 0:
                epoch_loss /= num_batches
                metrics['train_loss'].append(epoch_loss)
            
            # Validation
            if val_data:
                val_loss = self._evaluate(val_data, loss_fn)
                metrics['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}")
        
        # Move models back to evaluation mode
        self.user_model.eval()
        self.weight_model.eval()
        
        return metrics
    
    def _evaluate(self, val_data, loss_fn, batch_size=64):
        """
        Evaluate models on validation data.
        
        Args:
            val_data: Validation data
            loss_fn: Loss function
            batch_size: Batch size for evaluation
        
        Returns:
            Average validation loss
        """
        self.user_model.eval()
        self.weight_model.eval()
        
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                batch_loss = 0
                
                for example in batch:
                    user_id = example['user_id']
                    interactions = example['interactions']
                    target_embedding = torch.tensor(example['target_embedding'], dtype=torch.float32)
                    
                    # Get content embeddings
                    content_ids = [interaction['content_id'] for interaction in interactions]
                    content_vectors = self._get_content_vectors(content_ids)
                    
                    if not content_vectors:
                        continue
                    
                    # Prepare interaction data
                    interaction_embeddings = []
                    interaction_types = []
                    interaction_times = []
                    
                    for interaction in interactions:
                        content_id = interaction['content_id']
                        if content_id in content_vectors:
                            interaction_embeddings.append(content_vectors[content_id])
                            
                            int_type = self.interaction_types.get(
                                interaction['type'].lower(), 0
                            )
                            interaction_types.append(int_type)
                            
                            time_bucket = self._time_to_bucket(interaction.get('timestamp', datetime.now()))
                            interaction_times.append(time_bucket)
                    
                    if not interaction_embeddings:
                        continue
                    
                    # Convert to tensors
                    interaction_embeddings = torch.tensor(np.array(interaction_embeddings), dtype=torch.float32)
                    interaction_types = torch.tensor(interaction_types, dtype=torch.long)
                    interaction_times = torch.tensor(interaction_times, dtype=torch.long)
                    
                    # Forward pass
                    weights = self.weight_model(interaction_types, interaction_times)
                    user_embedding = self.user_model(interaction_embeddings, weights)
                    
                    # Compute loss
                    loss = loss_fn(user_embedding, target_embedding)
                    batch_loss += loss.item()
                
                if batch:
                    batch_loss /= len(batch)
                    val_loss += batch_loss
                    num_batches += 1
        
        self.user_model.train()
        self.weight_model.train()
        
        return val_loss / num_batches if num_batches > 0 else float('inf')

def prepare_training_data(interaction_data, target_window=7, history_window=30, min_interactions=5):
    """
    Prepare training data for the user embedding system.
    
    Args:
        interaction_data: List of user-content interactions
        target_window: Number of days in the target window (future interactions)
        history_window: Number of days in the history window (past interactions)
        min_interactions: Minimum number of interactions required
    
    Returns:
        List of training examples
    """
    # Group interactions by user
    user_interactions = defaultdict(list)
    for interaction in interaction_data:
        user_id = interaction['user_id']
        user_interactions[user_id].append(interaction)
    
    # Prepare training examples
    training_examples = []
    
    for user_id, interactions in user_interactions.items():
        # Sort interactions by timestamp
        interactions.sort(key=lambda x: x.get('timestamp', datetime.min))
        
        # Only consider users with enough interactions
        if len(interactions) < min_interactions:
            continue
        
        # Calculate cutoff date for history/target split
        latest_interaction = interactions[-1].get('timestamp', datetime.now())
        if isinstance(latest_interaction, str):
            latest_interaction = datetime.fromisoformat(latest_interaction)
        
        history_cutoff = latest_interaction - datetime.timedelta(days=target_window)
        history_start = history_cutoff - datetime.timedelta(days=history_window)
        
        # Split interactions into history and target
        history = []
        target = []
        
        for interaction in interactions:
            ts = interaction.get('timestamp', datetime.min)
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            
            if ts < history_cutoff and ts >= history_start:
                history.append(interaction)
            elif ts >= history_cutoff:
                target.append(interaction)
        
        # Only include users with both history and target interactions
        if not history or not target:
            continue
        
        # Create example
        example = {
            'user_id': user_id,
            'interactions': history,
            'target_interactions': target,
            'target_embedding': None  # To be filled
        }
        
        training_examples.append(example)
    
    return training_examples

def main():
    parser = argparse.ArgumentParser(description="User embedding system for content recommendations")
    
    parser.add_argument("--action", type=str, required=True, choices=[
        "update_users", "train", "recommend", "similar_users"
    ], help="Action to perform")
    
    parser.add_argument("--interactions-file", type=str,
                       help="Path to interactions data file (JSON or parquet)")
    
    parser.add_argument("--output-file", type=str,
                       help="Path to output file")
    
    parser.add_argument("--model-path", type=str, default="models/user_embedding_model.pt",
                       help="Path to save/load model")
    
    parser.add_argument("--qdrant-url", type=str, default="http://localhost",
                       help="URL of Qdrant service")
    
    parser.add_argument("--qdrant-api-key", type=str, default=None,
                       help="API key for Qdrant service")
    
    parser.add_argument("--content-collection", type=str, default="images",
                       help="Name of content collection in Qdrant")
    
    parser.add_argument("--user-collection", type=str, default="users",
                       help="Name of user collection in Qdrant")
    
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for training")
    
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate for training")
    
    parser.add_argument("--user-id", type=str,
                       help="User ID for recommendation or similar user actions")
    
    parser.add_argument("--num-recommendations", type=int, default=10,
                       help="Number of recommendations to generate")
    
    args = parser.parse_args()
    
    # Initialize user embedding system
    user_system = UserEmbeddingSystem(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        content_collection=args.content_collection,
        user_collection=args.user_collection,
        model_path=args.model_path if args.action != "train" else None
    )
    
    # Perform action
    if args.action == "update_users":
        if not args.interactions_file:
            print("Error: interactions-file is required for update_users action")
            return
        
        # Load interactions
        if args.interactions_file.endswith('.json'):
            with open(args.interactions_file, 'r') as f:
                interactions_data = json.load(f)
        else:
            interactions_data = wr.s3.read_parquet(args.interactions_file).to_dict('records')
        
        # Group by user
        user_interactions = defaultdict(list)
        for interaction in interactions_data:
            user_id = interaction['user_id']
            user_interactions[user_id].append(interaction)
        
        # Update users
        user_system.batch_update_users(user_interactions)
        
        print(f"Updated embeddings for {len(user_interactions)} users")
    
    elif args.action == "train":
        if not args.interactions_file:
            print("Error: interactions-file is required for train action")
            return
        
        # Load interactions
        if args.interactions_file.endswith('.json'):
            with open(args.interactions_file, 'r') as f:
                interactions_data = json.load(f)
        else:
            interactions_data = wr.s3.read_parquet(args.interactions_file).to_dict('records')
        
        # Prepare training data
        training_data = prepare_training_data(interactions_data)
        
        # Split into train/val
        np.random.shuffle(training_data)
        split_idx = int(len(training_data) * 0.8)
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        print(f"Training with {len(train_data)} examples, validating with {len(val_data)} examples")
        
        # Train model
        metrics = user_system.train(
            train_data=train_data,
            val_data=val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate
        )
        
        # Save model
        user_system.save_models(args.model_path)
        
        # Save metrics if output file specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    elif args.action == "recommend":
        if not args.user_id:
            print("Error: user-id is required for recommend action")
            return
        
        # Generate recommendations
        recommendations = user_system.recommend_for_user(
            user_id=args.user_id,
            k=args.num_recommendations
        )
        
        # Print or save recommendations
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
        else:
            print(f"Recommendations for user {args.user_id}:")
            for i, rec in enumerate(recommendations):
                print(f"{i+1}. ID: {rec['id']}, Score: {rec['score']:.4f}")
                print(f"   Description: {rec['description'][:100]}...")
                print()
    
    elif args.action == "similar_users":
        if not args.user_id:
            print("Error: user-id is required for similar_users action")
            return
        
        # Find similar users
        similar_users = user_system.find_similar_users(
            user_id=args.user_id,
            k=args.num_recommendations
        )
        
        # Print or save results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(similar_users, f, indent=2)
        else:
            print(f"Users similar to {args.user_id}:")
            for i, user in enumerate(similar_users):
                print(f"{i+1}. User ID: {user['user_id']}, Similarity: {user['similarity']:.4f}")

if __name__ == "__main__":
    main() 