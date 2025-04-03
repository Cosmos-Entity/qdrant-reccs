import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import random
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from qdrant_client import QdrantClient
import tarfile


class UserTower(nn.Module):
    """
    User tower for encoding user histories into embeddings.
    """
    def __init__(self, embedding_dim: int = 512, max_seq_len: int = 50, 
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.2):
        super(UserTower, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Positional embedding for sequence modeling
        self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection layer
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization for the output
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, item_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate user embeddings from interaction history.
        
        Args:
            item_embeddings: Sequence of item embeddings the user has interacted with
            attention_mask: Mask for padding in sequences
            
        Returns:
            User embedding
        """
        batch_size, seq_len, _ = item_embeddings.size()
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=item_embeddings.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.positional_embedding(positions)
        
        # Add positional embeddings to item embeddings
        x = item_embeddings + pos_emb
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply transformer encoder with masking
        x = self.transformer_encoder(x, src_key_padding_mask=(attention_mask == 0))
        
        # Get the embedding of the last non-padding item
        last_token_idx = attention_mask.sum(dim=1).long() - 1
        batch_indices = torch.arange(batch_size, device=item_embeddings.device)
        user_repr = x[batch_indices, last_token_idx, :]
        
        # Apply projection and normalization
        user_repr = self.projection(user_repr)
        user_repr = self.layer_norm(user_repr)
        
        return user_repr

class ItemTower(nn.Module):
    """
    Item tower for encoding items into embeddings.
    """
    def __init__(self, embedding_dim: int = 512, num_layers: int = 2, dropout: float = 0.1):
        super(ItemTower, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Fully connected layers for processing item features
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, item_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate item embeddings.
        
        Args:
            item_features: Raw item features (e.g., from ThreeVectorEncoder)
            
        Returns:
            Item embedding
        """
        return self.fc_layers(item_features)

class TwoTowerModel(nn.Module):
    """
    Two-tower model for collaborative filtering that jointly learns
    user and item representations.
    """
    def __init__(self, embedding_dim: int = 512, max_seq_len: int = 50, 
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.2,
                 temperature: float = 0.07):
        super(TwoTowerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # User tower
        self.user_tower = UserTower(
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Item tower
        self.item_tower = ItemTower(
            embedding_dim=embedding_dim,
            num_layers=2,
            dropout=dropout
        )
    
    def forward(self, user_histories: torch.Tensor, user_masks: torch.Tensor,
                items: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both towers.
        
        Args:
            user_histories: Batch of user interaction histories
            user_masks: Attention masks for user histories
            items: Batch of items to encode
            
        Returns:
            Dictionary with user and item embeddings
        """
        # Encode users
        user_embeddings = self.user_tower(user_histories, user_masks)
        
        # Encode items
        item_embeddings = self.item_tower(items)
        
        return {
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings
        }
    
    def compute_similarity(self, user_embeddings: torch.Tensor, 
                          item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between users and items.
        
        Args:
            user_embeddings: User embeddings
            item_embeddings: Item embeddings
            
        Returns:
            Similarity scores
        """
        # Compute dot product similarity
        if user_embeddings.dim() == 2 and item_embeddings.dim() == 2:
            # Compute similarity matrix between all users and items
            similarity = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1))
        else:
            # Compute similarity for corresponding pairs
            similarity = torch.sum(user_embeddings * item_embeddings, dim=-1)
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        return similarity
    
    def compute_contrastive_loss(self, user_embeddings: torch.Tensor,
                                positive_items: torch.Tensor,
                                negative_items: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss for training.
        
        Args:
            user_embeddings: User embeddings
            positive_items: Items the users have positively interacted with
            negative_items: Negative items for contrastive learning
            
        Returns:
            Contrastive loss value
        """
        batch_size = user_embeddings.size(0)
        
        # Encode items
        positive_embeddings = self.item_tower(positive_items)
        negative_embeddings = self.item_tower(negative_items)
        
        # Compute positive similarities
        pos_similarity = torch.sum(user_embeddings * positive_embeddings, dim=-1) / self.temperature
        
        # Compute negative similarities
        neg_similarity = torch.sum(user_embeddings.unsqueeze(1) * negative_embeddings, dim=-1) / self.temperature
        
        # Combine positive and negative similarities
        logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # First item is positive
        
        # Cross-entropy loss
        loss = nn.functional.cross_entropy(logits, labels)
        
        return loss
    
    def generate_embeddings(self, user_histories: Optional[torch.Tensor] = None,
                           user_masks: Optional[torch.Tensor] = None,
                           items: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate embeddings for users or items.
        
        Args:
            user_histories: Optional batch of user histories
            user_masks: Optional attention masks for user histories
            items: Optional batch of items
            
        Returns:
            Dictionary with generated embeddings
        """
        results = {}
        
        # Generate user embeddings if histories provided
        if user_histories is not None and user_masks is not None:
            results['user_embeddings'] = self.user_tower(user_histories, user_masks)
        
        # Generate item embeddings if items provided
        if items is not None:
            results['item_embeddings'] = self.item_tower(items)
        
        return results
    
    def get_user_embedding(self, history_embeddings: torch.Tensor, 
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get user embedding from interaction history.
        
        Args:
            history_embeddings: User interaction history embeddings
            attention_mask: Attention mask for padding
            
        Returns:
            User embedding
        """
        return self.user_tower(history_embeddings, attention_mask)
    
    def get_item_embedding(self, item_features: torch.Tensor) -> torch.Tensor:
        """
        Get item embedding from features.
        
        Args:
            item_features: Item features
            
        Returns:
            Item embedding
        """
        return self.item_tower(item_features)

class InteractionDataset(Dataset):
    """
    Dataset for training the two-tower model.
    """
    def __init__(
        self, 
        user_item_interactions, 
        user_histories,
        item_embeddings,
        user_metadata=None,
        item_metadata=None,
        negative_samples=4
    ):
        """
        Initialize the dataset.
        
        Args:
            user_item_interactions: List of (user_id, item_id, label) tuples
            user_histories: Dict mapping user_id to average history embedding
            item_embeddings: Dict mapping item_id to (image_embedding, text_embedding)
            user_metadata: Optional dict mapping user_id to metadata features
            item_metadata: Optional dict mapping item_id to metadata features
            negative_samples: Number of negative samples per positive example
        """
        self.user_item_interactions = user_item_interactions
        self.user_histories = user_histories
        self.item_embeddings = item_embeddings
        self.user_metadata = user_metadata
        self.item_metadata = item_metadata
        self.negative_samples = negative_samples
        
        # Create set of all items for negative sampling
        self.all_items = list(item_embeddings.keys())
        
        # Map each user to their positive items for efficient negative sampling
        self.user_to_items = defaultdict(set)
        for user_id, item_id, _ in user_item_interactions:
            self.user_to_items[user_id].add(item_id)
    
    def __len__(self):
        return len(self.user_item_interactions) * (1 + self.negative_samples)
    
    def __getitem__(self, idx):
        # Determine if this is a positive or negative sample
        interaction_idx = idx // (1 + self.negative_samples)
        is_negative = idx % (1 + self.negative_samples) != 0
        
        user_id, item_id, label = self.user_item_interactions[interaction_idx]
        
        # For negative samples, replace the item with a random one the user hasn't interacted with
        if is_negative:
            # Sample a random item the user hasn't interacted with
            negative_item = random.choice(self.all_items)
            while negative_item in self.user_to_items[user_id]:
                negative_item = random.choice(self.all_items)
            
            item_id = negative_item
            label = 0  # Negative label
        
        # Get user history embedding
        user_history = self.user_histories.get(user_id, torch.zeros(128))
        
        # Get item embeddings
        image_embedding, text_embedding = self.item_embeddings.get(
            item_id, 
            (torch.zeros(1024), torch.zeros(768))
        )
        
        # Get metadata if available
        user_metadata = self.user_metadata.get(user_id, None) if self.user_metadata else None
        item_metadata = self.item_metadata.get(item_id, None) if self.item_metadata else None
        
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
        
        # Use binary cross-entropy loss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )
        
        # Track metrics
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
            epochs: Number of epochs to train for
        
        Returns:
            Dictionary of training metrics
        """
        best_val_auc = 0.0
        best_model = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in progress_bar:
                # Move data to device
                user_history = batch['user_history'].float().to(self.device)
                image_embedding = batch['image_embedding'].float().to(self.device)
                text_embedding = batch['text_embedding'].float().to(self.device)
                label = batch['label'].float().to(self.device)
                
                # Get user and item metadata if available
                user_metadata = batch.get('user_metadata', None)
                if user_metadata is not None:
                    user_metadata = user_metadata.float().to(self.device)
                
                item_metadata = batch.get('item_metadata', None)
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
                
                # Compute loss
                loss = self.criterion(similarity, label)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({'loss': train_loss / (progress_bar.n + 1)})
            
            avg_train_loss = train_loss / len(self.train_loader)
            self.metrics['train_loss'].append(avg_train_loss)
            
            # Validation
            if self.val_loader:
                val_loss, val_auc, val_ap = self.evaluate()
                self.metrics['val_loss'].append(val_loss)
                self.metrics['val_auc'].append(val_auc)
                self.metrics['val_ap'].append(val_ap)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val AUC: {val_auc:.4f}, "
                      f"Val AP: {val_ap:.4f}")
                
                # Update learning rate scheduler
                self.scheduler.step(val_auc)
                
                # Save best model
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model = self.model.state_dict()
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Load best model if available
        if best_model:
            self.model.load_state_dict(best_model)
        
        return self.metrics
    
    def evaluate(self):
        """
        Evaluate the model on the validation set.
        
        Returns:
            Tuple of (loss, AUC, average precision)
        """
        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            for batch in progress_bar:
                # Move data to device
                user_history = batch['user_history'].float().to(self.device)
                image_embedding = batch['image_embedding'].float().to(self.device)
                text_embedding = batch['text_embedding'].float().to(self.device)
                label = batch['label'].float().to(self.device)
                
                # Get user and item metadata if available
                user_metadata = batch.get('user_metadata', None)
                if user_metadata is not None:
                    user_metadata = user_metadata.float().to(self.device)
                
                item_metadata = batch.get('item_metadata', None)
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
                
                # Compute loss
                loss = self.criterion(similarity, label)
                val_loss += loss.item()
                
                # Store predictions and labels for metrics
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(torch.sigmoid(similarity).cpu().numpy())
                
                progress_bar.set_postfix({'loss': val_loss / (progress_bar.n + 1)})
        
        avg_val_loss = val_loss / len(self.val_loader)
        val_auc = roc_auc_score(all_labels, all_preds)
        val_ap = average_precision_score(all_labels, all_preds)
        
        return avg_val_loss, val_auc, val_ap
    
    def save_model(self, path):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics
        }, path)
        print(f"Model saved to {path}")
    
    def plot_metrics(self, save_path=None):
        """
        Plot training metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        if 'val_loss' in self.metrics and self.metrics['val_loss']:
            plt.plot(self.metrics['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Plot validation metrics
        if 'val_auc' in self.metrics and self.metrics['val_auc']:
            plt.subplot(2, 1, 2)
            plt.plot(self.metrics['val_auc'], label='AUC')
            plt.plot(self.metrics['val_ap'], label='AP')
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.legend()
            plt.title('Validation Metrics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Metrics plot saved to {save_path}")
        
        plt.show()

class EmbeddingProcessor:
    """
    Process embeddings for the two-tower model.
    """
    def __init__(
        self,
        qdrant_url="http://localhost",
        user_collection="users",
        item_collection="elements",
        qdrant_api_key=None
    ):
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.user_collection = user_collection
        self.item_collection = item_collection
    
    def get_item_embeddings(self, item_ids):
        """
        Get item embeddings from Qdrant.
        
        Args:
            item_ids: List of item IDs
        
        Returns:
            Dictionary mapping item IDs to (image_embedding, text_embedding) tuples
        """
        try:
            # Retrieve items from Qdrant
            items = self.qdrant_client.retrieve(
                collection_name=self.item_collection,
                ids=item_ids,
                with_payload=True
            )
            
            item_embeddings = {}
            for item in items:
                item_id = item.id
                
                # Get embeddings from vector and payload
                image_embedding = torch.tensor(item.payload.get('image_vector', item.vector))
                
                # Get text embedding if available, otherwise use zeros
                text_embedding = item.payload.get('text_vector', None)
                if text_embedding is None:
                    # Check for description to generate text embedding
                    description = item.payload.get('description', '')
                    if description:
                        # In a real system, you would generate a text embedding here
                        # For now, use zeros
                        text_embedding = torch.zeros(768)
                    else:
                        text_embedding = torch.zeros(768)
                else:
                    text_embedding = torch.tensor(text_embedding)
                
                item_embeddings[item_id] = (image_embedding, text_embedding)
            
            return item_embeddings
        
        except Exception as e:
            print(f"Error retrieving item embeddings: {e}")
            return {}
    
    def get_user_histories(self, user_item_data):
        """
        Get user history embeddings.
        
        Args:
            user_item_data: Dictionary mapping user IDs to lists of item IDs
        
        Returns:
            Dictionary mapping user IDs to average history embeddings
        """
        user_histories = {}
        
        # Get all unique item IDs
        all_item_ids = set()
        for items in user_item_data.values():
            all_item_ids.update(items)
        
        # Get item embeddings
        item_embeddings = self.get_item_embeddings(list(all_item_ids))
        
        # Compute average embedding for each user's history
        for user_id, items in user_item_data.items():
            # Get embeddings for items in the user's history
            history_embeddings = []
            for item_id in items:
                if item_id in item_embeddings:
                    # Use image embedding as the item representation
                    image_embedding, _ = item_embeddings[item_id]
                    history_embeddings.append(image_embedding)
            
            if history_embeddings:
                # Compute average embedding
                avg_embedding = torch.stack(history_embeddings).mean(dim=0)
                user_histories[user_id] = avg_embedding
        
        return user_histories
    
    def prepare_training_data(self, user_item_data, val_fraction=0.2):
        """
        Prepare training data for the two-tower model.
        
        Args:
            user_item_data: Dictionary mapping user IDs to lists of item IDs
            val_fraction: Fraction of data to use for validation
        
        Returns:
            Tuple of (train_interactions, val_interactions, user_histories, item_embeddings)
        """
        # Create positive interactions
        interactions = []
        for user_id, items in user_item_data.items():
            for item_id in items:
                interactions.append((user_id, item_id, 1))  # Positive label
        
        # Shuffle interactions
        random.shuffle(interactions)
        
        # Split into train and validation
        split_idx = int(len(interactions) * (1 - val_fraction))
        train_interactions = interactions[:split_idx]
        val_interactions = interactions[split_idx:]
        
        # Get user histories
        user_histories = self.get_user_histories(user_item_data)
        
        # Get all unique item IDs
        all_item_ids = set()
        for user_id, items in user_item_data.items():
            all_item_ids.update(items)
        
        # Get item embeddings
        item_embeddings = self.get_item_embeddings(list(all_item_ids))
        
        return train_interactions, val_interactions, user_histories, item_embeddings
    
    def create_data_loaders(
        self,
        train_interactions,
        val_interactions,
        user_histories,
        item_embeddings,
        batch_size=32,
        negative_samples=4,
        user_metadata=None,
        item_metadata=None
    ):
        """
        Create DataLoader objects for training and validation.
        
        Args:
            train_interactions: List of (user_id, item_id, label) tuples for training
            val_interactions: List of (user_id, item_id, label) tuples for validation
            user_histories: Dict mapping user IDs to history embeddings
            item_embeddings: Dict mapping item IDs to embeddings
            batch_size: Batch size for DataLoader
            negative_samples: Number of negative samples per positive example
            user_metadata: Optional dict mapping user IDs to metadata
            item_metadata: Optional dict mapping item IDs to metadata
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create datasets
        train_dataset = InteractionDataset(
            train_interactions,
            user_histories,
            item_embeddings,
            user_metadata,
            item_metadata,
            negative_samples
        )
        
        val_dataset = InteractionDataset(
            val_interactions,
            user_histories,
            item_embeddings,
            user_metadata,
            item_metadata,
            negative_samples
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def upload_embeddings_to_qdrant(self, model, user_item_data, output_collection=None):
        """
        Generate embeddings using the trained model and upload them to Qdrant.
        
        Args:
            model: Trained two-tower model
            user_item_data: Dictionary mapping user IDs to lists of item IDs
            output_collection: Name of the collection to upload embeddings to
                              (defaults to user_collection and item_collection)
        
        Returns:
            Number of user and item embeddings uploaded
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Get user histories
        user_histories = self.get_user_histories(user_item_data)
        
        # Get all unique item IDs
        all_item_ids = set()
        for user_id, items in user_item_data.items():
            all_item_ids.update(items)
        
        # Get item embeddings
        raw_item_embeddings = self.get_item_embeddings(list(all_item_ids))
        
        # Process user embeddings
        user_embeddings = []
        print("Generating user embeddings...")
        for user_id, history_embedding in tqdm(user_histories.items()):
            # Get user embedding
            with torch.no_grad():
                user_embedding = model.get_user_embedding(
                    history_embedding.unsqueeze(0).to(device)
                )
            
            # Create point for Qdrant
            user_embeddings.append({
                'id': user_id,
                'vector': user_embedding.squeeze().cpu().numpy().tolist(),
                'payload': {
                    'user_id': user_id,
                    'history_length': len(user_item_data[user_id])
                }
            })
        
        # Process item embeddings
        item_embeddings = []
        print("Generating item embeddings...")
        for item_id, (image_embedding, text_embedding) in tqdm(raw_item_embeddings.items()):
            # Get item embedding
            with torch.no_grad():
                item_embedding = model.get_item_embedding(
                    image_embedding.unsqueeze(0).to(device),
                    text_embedding.unsqueeze(0).to(device)
                )
            
            # Create point for Qdrant
            item_embeddings.append({
                'id': item_id,
                'vector': item_embedding.squeeze().cpu().numpy().tolist(),
                'payload': {
                    'item_id': item_id
                }
            })
        
        # Upload user embeddings
        user_output = output_collection or f"{self.user_collection}_two_tower"
        self._ensure_collection_exists(user_output, model.embedding_dim)
        
        batch_size = 100
        for i in range(0, len(user_embeddings), batch_size):
            batch = user_embeddings[i:i+batch_size]
            self.qdrant_client.upsert(
                collection_name=user_output,
                points=batch
            )
        
        # Upload item embeddings
        item_output = output_collection or f"{self.item_collection}_two_tower"
        if item_output != user_output:
            self._ensure_collection_exists(item_output, model.embedding_dim)
        
        for i in range(0, len(item_embeddings), batch_size):
            batch = item_embeddings[i:i+batch_size]
            self.qdrant_client.upsert(
                collection_name=item_output,
                points=batch
            )
        
        print(f"Uploaded {len(user_embeddings)} user embeddings to {user_output}")
        print(f"Uploaded {len(item_embeddings)} item embeddings to {item_output}")
        
        return len(user_embeddings), len(item_embeddings)
    
    def _ensure_collection_exists(self, collection_name, vector_dim):
        """
        Ensure the collection exists in Qdrant.
        """
        try:
            self.qdrant_client.get_collection(collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": vector_dim, "distance": "Cosine"}
            )
            print(f"Created new collection: {collection_name}")

def process_user_snapshot_data(snapshot_data_path):
    """
    Process user data from the extracted snapshot.
    
    Args:
        snapshot_data_path: Path to the extracted snapshot data directory
    
    Returns:
        Dictionary mapping user_id to element_ids
    """
    user_data = {}
    segments_dir = os.path.join(snapshot_data_path, "0/segments")
    
    # Check if directory exists
    if not os.path.exists(segments_dir):
        print(f"Segments directory not found: {segments_dir}")
        return user_data
    
    # Process each segment file
    for filename in os.listdir(segments_dir):
        if filename.endswith('.tar'):
            segment_path = os.path.join(segments_dir, filename)
            
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
    
    return user_data

def main():
    parser = argparse.ArgumentParser(description="Two-tower model for learning user and item embeddings")
    
    parser.add_argument("--user-snapshot", type=str, default="Cosmos Users Snapshot Apr 1 2025.snapshot",
                      help="Path to user snapshot file")
    
    parser.add_argument("--extracted-dir", type=str, default="extracted_users_data",
                      help="Directory for extracted snapshot data")
    
    parser.add_argument("--model-output", type=str, default="models/two_tower_model.pt",
                      help="Path to save the trained model")
    
    parser.add_argument("--plot-output", type=str, default="outputs/training_metrics.png",
                      help="Path to save training metrics plot")
    
    parser.add_argument("--embedding-dim", type=int, default=128,
                      help="Dimension of the embedding space")
    
    parser.add_argument("--batch-size", type=int, default=64,
                      help="Batch size for training")
    
    parser.add_argument("--epochs", type=int, default=10,
                      help="Number of epochs to train")
    
    parser.add_argument("--learning-rate", type=float, default=0.001,
                      help="Learning rate for optimizer")
    
    parser.add_argument("--negative-samples", type=int, default=4,
                      help="Number of negative samples per positive example")
    
    parser.add_argument("--qdrant-url", type=str, default="http://localhost",
                      help="URL of Qdrant service")
    
    parser.add_argument("--qdrant-api-key", type=str, default=None,
                      help="API key for Qdrant service")
    
    parser.add_argument("--user-collection", type=str, default="users",
                      help="Name of user collection in Qdrant")
    
    parser.add_argument("--item-collection", type=str, default="elements",
                      help="Name of item collection in Qdrant")
    
    parser.add_argument("--output-collection", type=str, default=None,
                      help="Name of collection to upload embeddings to")
    
    args = parser.parse_args()
    
    # Process user snapshot data
    if not os.path.exists(args.extracted_dir):
        # Extract the snapshot first
        try:
            from extract_snapshot_data import extract_snapshot_data
            extract_snapshot_data(args.user_snapshot, args.extracted_dir)
        except ImportError:
            print("extract_snapshot_data.py not found. Please extract the snapshot manually.")
            return
    
    user_data = process_user_snapshot_data(args.extracted_dir)
    print(f"Processed data for {len(user_data)} users")
    
    # Initialize embedding processor
    processor = EmbeddingProcessor(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        user_collection=args.user_collection,
        item_collection=args.item_collection
    )
    
    # Prepare training data
    train_interactions, val_interactions, user_histories, item_embeddings = processor.prepare_training_data(user_data)
    print(f"Prepared {len(train_interactions)} training and {len(val_interactions)} validation interactions")
    print(f"Got history embeddings for {len(user_histories)} users")
    print(f"Got embeddings for {len(item_embeddings)} items")
    
    # Create data loaders
    train_loader, val_loader = processor.create_data_loaders(
        train_interactions,
        val_interactions,
        user_histories,
        item_embeddings,
        batch_size=args.batch_size,
        negative_samples=args.negative_samples
    )
    
    # Determine input dimensions from data
    sample_user_history = next(iter(user_histories.values()))
    sample_image_embedding, sample_text_embedding = next(iter(item_embeddings.values()))
    
    image_embedding_dim = sample_image_embedding.shape[0]
    text_embedding_dim = sample_text_embedding.shape[0]
    
    # Initialize model
    model = TwoTowerModel(
        embedding_dim=args.embedding_dim,
        image_embedding_dim=image_embedding_dim,
        text_embedding_dim=text_embedding_dim
    )
    
    # Initialize trainer
    trainer = TwoTowerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate
    )
    
    # Train model
    print("Starting training...")
    metrics = trainer.train(epochs=args.epochs)
    
    # Save model
    trainer.save_model(args.model_output)
    
    # Plot metrics
    trainer.plot_metrics(args.plot_output)
    
    # Upload embeddings to Qdrant
    print("Generating and uploading embeddings...")
    user_count, item_count = processor.upload_embeddings_to_qdrant(
        model=model,
        user_item_data=user_data,
        output_collection=args.output_collection
    )
    
    print(f"Successfully generated and uploaded {user_count} user and {item_count} item embeddings")
    print("Training complete!")

if __name__ == "__main__":
    main() 