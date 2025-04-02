import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import random
from tqdm import tqdm

# Setup logging
logger = logging.getLogger(__name__)

class UserTower(nn.Module):
    """
    Neural network tower for user embeddings.
    """
    def __init__(
        self, 
        embedding_dim: int = 128, 
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        user_metadata_dim: int = 0
    ):
        super(UserTower, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.has_metadata = user_metadata_dim > 0
        
        # Input dimension calculation
        input_dim = embedding_dim
        if self.has_metadata:
            input_dim += user_metadata_dim
        
        # Layers
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, embedding_dim))
        layers.append(nn.LayerNorm(embedding_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, metadata=None):
        """
        Forward pass.
        
        Args:
            x: User history embeddings [batch_size, embedding_dim]
            metadata: Optional metadata [batch_size, metadata_dim]
        
        Returns:
            User embedding
        """
        if self.has_metadata and metadata is not None:
            # Concatenate history embeddings with metadata
            x = torch.cat([x, metadata], dim=1)
        
        return self.model(x)

class ItemTower(nn.Module):
    """
    Neural network tower for item embeddings.
    """
    def __init__(
        self, 
        embedding_dim: int = 128, 
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        image_embedding_dim: int = 1024,
        text_embedding_dim: int = 768,
        metadata_dim: int = 0
    ):
        super(ItemTower, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.has_metadata = metadata_dim > 0
        
        # Process image embeddings
        self.image_projection = nn.Linear(image_embedding_dim, embedding_dim)
        
        # Process text embeddings
        self.text_projection = nn.Linear(text_embedding_dim, embedding_dim)
        
        # Process metadata if available
        if self.has_metadata:
            self.metadata_projection = nn.Linear(metadata_dim, embedding_dim)
        
        # Input dimension for main model
        input_dim = embedding_dim * (2 + (1 if self.has_metadata else 0))
        
        # Layers
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, embedding_dim))
        layers.append(nn.LayerNorm(embedding_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, image_embedding, text_embedding, metadata=None):
        """
        Forward pass.
        
        Args:
            image_embedding: Image embeddings [batch_size, image_embedding_dim]
            text_embedding: Text embeddings [batch_size, text_embedding_dim]
            metadata: Optional metadata [batch_size, metadata_dim]
        
        Returns:
            Item embedding
        """
        # Project each embedding type
        image_proj = self.image_projection(image_embedding)
        text_proj = self.text_projection(text_embedding)
        
        # Combine embeddings
        if self.has_metadata and metadata is not None:
            metadata_proj = self.metadata_projection(metadata)
            combined = torch.cat([image_proj, text_proj, metadata_proj], dim=1)
        else:
            combined = torch.cat([image_proj, text_proj], dim=1)
        
        return self.model(combined)

class TwoTowerModel(nn.Module):
    """
    Two-tower model for jointly learning user and item embeddings.
    """
    def __init__(
        self, 
        embedding_dim: int = 128,
        user_hidden_dims: List[int] = [256, 128],
        item_hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        user_metadata_dim: int = 0,
        image_embedding_dim: int = 1024,
        text_embedding_dim: int = 768,
        item_metadata_dim: int = 0,
        temperature: float = 0.05
    ):
        super(TwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # Create user tower
        self.user_tower = UserTower(
            embedding_dim=embedding_dim,
            hidden_dims=user_hidden_dims,
            dropout=dropout,
            user_metadata_dim=user_metadata_dim
        )
        
        # Create item tower
        self.item_tower = ItemTower(
            embedding_dim=embedding_dim,
            hidden_dims=item_hidden_dims,
            dropout=dropout,
            image_embedding_dim=image_embedding_dim,
            text_embedding_dim=text_embedding_dim,
            metadata_dim=item_metadata_dim
        )
    
    def forward(
        self, 
        user_history, 
        image_embedding, 
        text_embedding,
        user_metadata=None,
        item_metadata=None
    ):
        """
        Forward pass.
        
        Args:
            user_history: User history embeddings [batch_size, embedding_dim]
            image_embedding: Image embeddings [batch_size, image_embedding_dim]
            text_embedding: Text embeddings [batch_size, text_embedding_dim]
            user_metadata: Optional user metadata [batch_size, user_metadata_dim]
            item_metadata: Optional item metadata [batch_size, item_metadata_dim]
        
        Returns:
            User embeddings, item embeddings, and similarity scores
        """
        # Get user embeddings
        user_embedding = self.user_tower(user_history, user_metadata)
        
        # Get item embeddings
        item_embedding = self.item_tower(image_embedding, text_embedding, item_metadata)
        
        # Compute similarity scores (cosine similarity with temperature scaling)
        user_norm = torch.nn.functional.normalize(user_embedding, p=2, dim=1)
        item_norm = torch.nn.functional.normalize(item_embedding, p=2, dim=1)
        
        # Compute batch dot product
        similarity = torch.sum(user_norm * item_norm, dim=1) / self.temperature
        
        return user_embedding, item_embedding, similarity
    
    def get_user_embedding(self, user_history, user_metadata=None):
        """Get user embedding from user tower."""
        with torch.no_grad():
            return self.user_tower(user_history, user_metadata)
    
    def get_item_embedding(self, image_embedding, text_embedding, item_metadata=None):
        """Get item embedding from item tower."""
        with torch.no_grad():
            return self.item_tower(image_embedding, text_embedding, item_metadata)

class InteractionDataset(Dataset):
    """
    Dataset for training the two-tower model with user-item interactions.
    """
    def __init__(
        self, 
        interactions, 
        user_histories,
        item_embeddings,
        user_metadata=None,
        item_metadata=None,
        negative_samples=4
    ):
        """
        Initialize dataset.
        
        Args:
            interactions: List of (user_id, item_id, label) tuples
            user_histories: Dict mapping user_id to history embedding
            item_embeddings: Dict mapping item_id to (image_embedding, text_embedding)
            user_metadata: Optional dict mapping user_id to metadata
            item_metadata: Optional dict mapping item_id to metadata
            negative_samples: Number of negative samples per positive sample
        """
        self.interactions = interactions
        self.user_histories = user_histories
        self.item_embeddings = item_embeddings
        self.user_metadata = user_metadata
        self.item_metadata = item_metadata
        self.negative_samples = negative_samples
        
        # Create set of all items for negative sampling
        self.all_items = list(item_embeddings.keys())
        
        # Map users to their interacted items for efficient negative sampling
        self.user_items = defaultdict(set)
        for user_id, item_id, _ in interactions:
            self.user_items[user_id].add(item_id)
    
    def __len__(self):
        return len(self.interactions) * (1 + self.negative_samples)
    
    def __getitem__(self, idx):
        """
        Get a dataset item.
        
        For each positive sample, we also generate negative_samples negative samples.
        """
        # Determine if this is a positive or negative sample
        interaction_idx = idx // (1 + self.negative_samples)
        is_negative = idx % (1 + self.negative_samples) != 0
        
        # Get the base interaction
        user_id, item_id, label = self.interactions[interaction_idx]
        
        # For negative samples, replace the item with a random one
        if is_negative:
            # Sample an item the user hasn't interacted with
            negative_item = random.choice(self.all_items)
            while negative_item in self.user_items[user_id]:
                negative_item = random.choice(self.all_items)
            
            item_id = negative_item
            label = 0  # Negative label
        
        # Get user history
        user_history = self.user_histories.get(user_id, torch.zeros(self.embedding_dim))
        
        # Get item embeddings
        image_embedding, text_embedding = self.item_embeddings.get(
            item_id, 
            (torch.zeros(self.image_embedding_dim), torch.zeros(self.text_embedding_dim))
        )
        
        # Get metadata if available
        user_metadata = None
        if self.user_metadata:
            user_metadata = self.user_metadata.get(user_id)
        
        item_metadata = None
        if self.item_metadata:
            item_metadata = self.item_metadata.get(item_id)
        
        return {
            'user_id': user_id,
            'item_id': item_id,
            'user_history': user_history,
            'image_embedding': image_embedding,
            'text_embedding': text_embedding,
            'user_metadata': user_metadata,
            'item_metadata': item_metadata,
            'label': label
        }

class TwoTowerTrainer:
    """
    Trainer for the two-tower model.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=0.001,
        weight_decay=0.0001,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )
        
        # Metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_ap': []
        }
    
    def train(self, epochs=10):
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            Training metrics
        """
        best_val_metric = 0
        best_model = None
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch()
            
            # Validation
            if self.val_loader:
                val_metrics = self._validate()
                val_loss = val_metrics['loss']
                val_auc = val_metrics['auc']
                
                # Update learning rate
                self.scheduler.step(val_auc)
                
                # Save best model
                if val_auc > best_val_metric:
                    best_val_metric = val_auc
                    best_model = {k: v.cpu() for k, v in self.model.state_dict().items()}
                
                # Log
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val AUC: {val_auc:.4f}")
                
                # Update metrics
                self.metrics['train_loss'].append(train_loss)
                self.metrics['val_loss'].append(val_loss)
                self.metrics['val_auc'].append(val_auc)
                self.metrics['val_ap'].append(val_metrics['ap'])
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
                self.metrics['train_loss'].append(train_loss)
        
        # Load best model if available
        if best_model:
            self.model.load_state_dict(best_model)
        
        return self.metrics
    
    def _train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move data to device
            user_history = batch['user_history'].float().to(self.device)
            image_embedding = batch['image_embedding'].float().to(self.device)
            text_embedding = batch['text_embedding'].float().to(self.device)
            labels = batch['label'].float().to(self.device)
            
            # Optional metadata
            user_metadata = batch.get('user_metadata')
            if user_metadata is not None:
                user_metadata = user_metadata.float().to(self.device)
            
            item_metadata = batch.get('item_metadata')
            if item_metadata is not None:
                item_metadata = item_metadata.float().to(self.device)
            
            # Forward pass
            _, _, similarity = self.model(
                user_history, 
                image_embedding, 
                text_embedding,
                user_metadata,
                item_metadata
            )
            
            # Loss
            loss = self.criterion(similarity, labels)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * len(labels)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader.dataset)
    
    def _validate(self):
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                user_history = batch['user_history'].float().to(self.device)
                image_embedding = batch['image_embedding'].float().to(self.device)
                text_embedding = batch['text_embedding'].float().to(self.device)
                labels = batch['label'].float().to(self.device)
                
                # Optional metadata
                user_metadata = batch.get('user_metadata')
                if user_metadata is not None:
                    user_metadata = user_metadata.float().to(self.device)
                
                item_metadata = batch.get('item_metadata')
                if item_metadata is not None:
                    item_metadata = item_metadata.float().to(self.device)
                
                # Forward pass
                _, _, similarity = self.model(
                    user_history, 
                    image_embedding, 
                    text_embedding,
                    user_metadata,
                    item_metadata
                )
                
                # Loss
                loss = self.criterion(similarity, labels)
                total_loss += loss.item() * len(labels)
                
                # Store predictions and labels for metrics
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(torch.sigmoid(similarity).cpu().numpy())
        
        # Compute metrics
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        metrics = {
            'loss': total_loss / len(self.val_loader.dataset)
        }
        
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_preds)
            metrics['ap'] = average_precision_score(all_labels, all_preds)
        except:
            metrics['auc'] = 0.5
            metrics['ap'] = 0.5
            logger.warning("Could not compute AUC/AP metrics")
        
        return metrics
    
    def save_model(self, path):
        """Save model to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics
        }, path)
        
        logger.info(f"Model saved to {path}")

# Helper functions for preparing data
def prepare_user_histories(user_data, content_vectors):
    """
    Prepare user history embeddings.
    
    Args:
        user_data: Dictionary mapping user_id to list of item_ids
        content_vectors: Dictionary mapping item_id to vector
        
    Returns:
        Dictionary mapping user_id to average history embedding
    """
    user_histories = {}
    
    for user_id, item_ids in user_data.items():
        # Get vectors for items the user has interacted with
        item_vectors = []
        for item_id in item_ids:
            if item_id in content_vectors:
                item_vectors.append(content_vectors[item_id])
        
        if item_vectors:
            # Average the item vectors
            avg_vector = np.mean(item_vectors, axis=0)
            user_histories[user_id] = torch.tensor(avg_vector, dtype=torch.float32)
    
    return user_histories

def prepare_interactions(user_data):
    """
    Prepare interaction data for training.
    
    Args:
        user_data: Dictionary mapping user_id to list of item_ids
        
    Returns:
        List of (user_id, item_id, label) tuples
    """
    interactions = []
    
    for user_id, item_ids in user_data.items():
        for item_id in item_ids:
            # Positive interaction
            interactions.append((user_id, item_id, 1))
    
    return interactions

def prepare_item_embeddings(item_ids, qdrant_client, collection_name):
    """
    Prepare item embeddings for training.
    
    Args:
        item_ids: List of item IDs
        qdrant_client: Qdrant client
        collection_name: Name of the collection
        
    Returns:
        Dictionary mapping item_id to (image_embedding, text_embedding)
    """
    item_embeddings = {}
    
    # Batch retrieval
    batch_size = 100
    for i in range(0, len(item_ids), batch_size):
        batch_ids = item_ids[i:i+batch_size]
        
        response = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=batch_ids,
            with_payload=True,
            with_vectors=True
        )
        
        for point in response:
            item_id = point.id
            
            # Primary vector is usually the combined or image vector
            image_vector = np.array(point.vector)
            
            # Look for text vector in payload
            text_vector = None
            if hasattr(point, 'payload') and point.payload:
                if 'vectors' in point.payload and 'text_vector' in point.payload['vectors']:
                    text_vector = np.array(point.payload['vectors']['text_vector'])
                elif 'description' in point.payload:
                    # In a real system, we would generate a text embedding here
                    # For this example, we'll use zeros
                    text_vector = np.zeros(768)
            
            # Default text vector if not found
            if text_vector is None:
                text_vector = np.zeros(768)
            
            # Store as tensors
            item_embeddings[item_id] = (
                torch.tensor(image_vector, dtype=torch.float32),
                torch.tensor(text_vector, dtype=torch.float32)
            )
    
    return item_embeddings

# Example usage
if __name__ == "__main__":
    # Test the model
    embedding_dim = 128
    image_dim = 1024
    text_dim = 768
    
    model = TwoTowerModel(
        embedding_dim=embedding_dim,
        image_embedding_dim=image_dim,
        text_embedding_dim=text_dim
    )
    
    # Sample batch
    batch_size = 2
    user_history = torch.randn(batch_size, embedding_dim)
    image_embedding = torch.randn(batch_size, image_dim)
    text_embedding = torch.randn(batch_size, text_dim)
    
    # Forward pass
    user_embed, item_embed, sim = model(user_history, image_embedding, text_embedding)
    
    print(f"User embedding shape: {user_embed.shape}")
    print(f"Item embedding shape: {item_embed.shape}")
    print(f"Similarity scores: {sim}") 