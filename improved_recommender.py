import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import awswrangler as wr
from torch.utils.data import DataLoader, Dataset
from qdrant_client import QdrantClient
import time
import logging
import os
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

# Import our improvement modules
from utils.query_transformation import QueryTransformer
from utils.three_vector_embedding import ThreeVectorEmbedding
from utils.user_embeddings import UserEmbeddingSystem
from utils.two_tower_model import TwoTowerModel, InteractionDataset, TwoTowerTrainer
from utils.advanced_search import AdvancedSearch, SearchType, MetadataFilter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the UserTransformer Model (modified SASRec with improvements)
class SASRec(nn.Module):
    def __init__(self, max_seq_len=50, embedding_dim=1024, num_layers=4, num_heads=8, dropout=0.2):
        super(SASRec, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Improved transformer encoder with layer normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Add normalization for improved training stability
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.size()
        
        # Create positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.positional_embedding(positions)
        
        # Add positional embeddings
        x = x + pos_emb
        x = self.dropout(x)
        
        # Apply transformer encoder with mask
        x = self.transformer_encoder(x, src_key_padding_mask=(mask == 0))
        
        # Get last token representation for user embedding
        last_token_idx = mask.sum(dim=1).long() - 1
        batch_indices = torch.arange(batch_size, device=x.device)
        user_repr = x[batch_indices, last_token_idx, :]
        
        # Apply final normalization
        user_repr = self.layer_norm(user_repr)
        
        return user_repr

# Define Dataset Class with improvements
class RecommendationDataset(Dataset):
    def __init__(
        self, 
        train_parquet, 
        val_parquet=None, 
        chunk_size=10000, 
        qdrant_url="http://localhost", 
        qdrant_collection="images", 
        qdrant_api_key=None,
        use_three_vector=True,
        use_query_transformation=True,
        max_seq_len=50
    ):
        self.train_parquet = train_parquet
        self.chunk_size = chunk_size
        self.qdrant_url = qdrant_url
        self.qdrant_collection = qdrant_collection
        self.qdrant_api_key = qdrant_api_key
        self.use_three_vector = use_three_vector
        self.use_query_transformation = use_query_transformation
        self.max_seq_len = max_seq_len
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Initialize query transformer if enabled
        if use_query_transformation:
            self.query_transformer = QueryTransformer()
        
        # Initialize three-vector embedder if enabled
        if use_three_vector:
            self.vector_embedder = ThreeVectorEmbedding(
                qdrant_url=qdrant_url,
                qdrant_collection=qdrant_collection,
                qdrant_api_key=qdrant_api_key
            )
            
        # Load validation data if provided
        if val_parquet:
            self.val_df = wr.s3.read_parquet(path=val_parquet).reset_index(drop=True)
            self.val_map = dict(zip(self.val_df['user_idx'], self.val_df['val']))
        else:
            self.val_map = {}
    
    def __iter__(self):
        # Initialize progress monitoring
        total_rows = 0
        processed_rows = 0
        start_time = time.time()
        
        # Read training data in chunks
        train_iter = wr.s3.read_parquet(path=self.train_parquet, chunked=True)
        chunk_count = 0
        
        for chunk in train_iter:
            chunk_count += 1
            chunk = chunk.reset_index(drop=True)
            total_rows += len(chunk)
            
            for idx, row in chunk.iterrows():
                processed_rows += 1
                
                # Log progress every 1000 rows
                if processed_rows % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_rows / elapsed if elapsed > 0 else 0
                    logger.info(f"Processing row {processed_rows}/{total_rows} (Chunk {chunk_count}, {rate:.2f} rows/sec)")
                
                # Process and yield the row
                yield self.process_row(row)
    
    def process_row(self, row):
        """Process a row with improvements: query transformation and three-vector embedding"""
        processed_row = row.copy()
        
        # Apply query transformation if enabled
        if self.use_query_transformation and 'query' in row:
            processed_row['query'] = self.query_transformer.preprocess_query(row['query'])
        
        # Apply three-vector embedding if enabled
        if self.use_three_vector and 'item_id' in row:
            # Get item vectors
            item_vectors = self.vector_embedder.get_item_vectors(row['item_id'])
            if item_vectors:
                processed_row['image_vector'] = item_vectors['image_vector']
                processed_row['text_vector'] = item_vectors['text_vector']
                processed_row['metadata_vector'] = item_vectors['metadata_vector']
        
        # Ensure training sequence is properly padded
        if 'train_seq' in processed_row:
            # Pad or truncate the sequence to max length
            train_seq = processed_row['train_seq']
            if len(train_seq) > self.max_seq_len:
                processed_row['train_seq_padded'] = train_seq[-self.max_seq_len:]
                processed_row['train_seq_mask'] = [1] * self.max_seq_len
            else:
                padding_len = self.max_seq_len - len(train_seq)
                processed_row['train_seq_padded'] = train_seq + [0] * padding_len
                processed_row['train_seq_mask'] = [1] * len(train_seq) + [0] * padding_len
        
        return processed_row

# Improved training function
def train_model(
    train_loader, 
    model, 
    optimizer, 
    scaler, 
    device, 
    epochs, 
    checkpoint_dir, 
    log_dir, 
    max_grad_norm, 
    qdrant_client, 
    qdrant_collection, 
    num_items,
    use_two_tower=True,  # Flag to enable two-tower model
    user_embedding_system=None  # Optional user embedding system
):
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Training metrics
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Two-tower model initialization if enabled
    two_tower_model = None
    if use_two_tower:
        two_tower_model = TwoTowerModel(
            embedding_dim=model.max_seq_len,
            image_embedding_dim=1024,
            text_embedding_dim=768
        ).to(device)
        two_tower_optimizer = optim.Adam(two_tower_model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # Process batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # Prepare data
            train_seq_padded = batch['train_seq_padded'].to(device)
            train_seq_mask = batch['train_seq_mask'].to(device)
            
            # Forward pass for main model (SASRec)
            user_embeddings = model(train_seq_padded, train_seq_mask)
            
            # Main model loss calculation and optimization
            if 'user_idx' in batch:
                user_idx = batch['user_idx'].to(device).float()
                loss = torch.nn.functional.mse_loss(user_embeddings, user_idx)
            else:
                # Alternative loss for sequence prediction
                loss = torch.nn.functional.cross_entropy(
                    user_embeddings, 
                    train_seq_padded[:, -1]
                )
            
            # Additional two-tower model training if enabled
            if use_two_tower and 'image_vector' in batch and 'text_vector' in batch:
                # Extract vectors
                image_vectors = batch['image_vector'].to(device)
                text_vectors = batch['text_vector'].to(device)
                
                # Train two-tower model
                two_tower_optimizer.zero_grad()
                _, _, similarity = two_tower_model(
                    user_embeddings,
                    image_vectors,
                    text_vectors
                )
                
                # Two-tower loss (assuming binary labels for now)
                labels = torch.ones(similarity.shape[0], device=device)
                two_tower_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    similarity, labels
                )
                
                # Combine losses
                loss = loss + two_tower_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        metrics['train_loss'].append(avg_epoch_loss)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        
        # Update user embeddings if system is provided
        if user_embedding_system is not None:
            logger.info("Updating user embeddings...")
            # This would extract user representations from the model and update the embedding system
            # Placeholder for actual implementation
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "model_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': metrics['train_loss'][-1],
    }, final_model_path)
    
    # Save training metrics
    metrics_path = os.path.join(log_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser()
    
    # Base parameters from original code
    parser.add_argument("--train-parquet", type=str, required=True)
    parser.add_argument("--val-parquet", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--use-amp", type=bool, default=True)
    parser.add_argument("--qdrant-url", type=str, default="http://localhost")
    parser.add_argument("--qdrant-api-key", type=str, default=None)
    parser.add_argument("--qdrant-collection", type=str, default="images")
    
    # New parameters for improvements
    parser.add_argument("--use-query-transformation", type=bool, default=True,
                        help="Use query transformation for preprocessing")
    parser.add_argument("--use-three-vector", type=bool, default=True,
                        help="Use three-vector architecture for embeddings")
    parser.add_argument("--use-user-embeddings", type=bool, default=True,
                        help="Use user embeddings in latent space")
    parser.add_argument("--use-two-tower", type=bool, default=True,
                        help="Use two-tower model for training")
    parser.add_argument("--use-advanced-search", type=bool, default=True,
                        help="Use advanced search techniques")
    
    # Additional parameters for new features
    parser.add_argument("--user-collection", type=str, default="users",
                        help="Qdrant collection for user embeddings")
    parser.add_argument("--embedding-dim", type=int, default=1024,
                        help="Dimension of embeddings")
    parser.add_argument("--search-type", type=str, choices=["vector", "text", "hybrid", "personalized"],
                        default="hybrid", help="Type of search to use")
    parser.add_argument("--vector-weight", type=float, default=0.7,
                        help="Weight for vector search in hybrid mode")
    parser.add_argument("--text-weight", type=float, default=0.3,
                        help="Weight for text search in hybrid mode")
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key, timeout=20)
    
    # Determine model dimensions
    # Read first chunk to calculate max item ID
    sample_chunk = wr.s3.read_parquet(path=args.train_parquet, chunked=True).__next__().reset_index(drop=True)
    train_max = sample_chunk['train_seq_padded'].apply(lambda s: max(s) if len(s) > 0 else 0).max()
    val_df = wr.s3.read_parquet(path=args.val_parquet)
    val_max = val_df['val'].max()
    num_items = int(max(train_max, val_max)) + 1
    logger.info(f"Computed num_items: {num_items}")
    
    # Initialize user embedding system if enabled
    user_embedding_system = None
    if args.use_user_embeddings:
        logger.info("Initializing user embedding system...")
        user_embedding_system = UserEmbeddingSystem(
            vector_dim=args.embedding_dim,
            qdrant_url=args.qdrant_url,
            qdrant_collection=args.qdrant_collection,
            user_collection=args.user_collection,
            qdrant_api_key=args.qdrant_api_key
        )
    
    # Initialize advanced search if enabled
    search_engine = None
    if args.use_advanced_search:
        logger.info("Initializing advanced search engine...")
        search_engine = AdvancedSearch(
            qdrant_url=args.qdrant_url,
            collection_name=args.qdrant_collection,
            api_key=args.qdrant_api_key,
            vector_weight=args.vector_weight,
            text_weight=args.text_weight
        )
    
    # Initialize dataset
    train_dataset = RecommendationDataset(
        args.train_parquet,
        args.val_parquet,
        args.chunk_size,
        args.qdrant_url,
        args.qdrant_collection,
        args.qdrant_api_key,
        use_three_vector=args.use_three_vector,
        use_query_transformation=args.use_query_transformation
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = SASRec(
        max_seq_len=50,
        embedding_dim=args.embedding_dim,
        num_layers=4,
        num_heads=8,
        dropout=0.2
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Initialize AMP scaler if using mixed precision
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.use_amp) else None
    
    # Train the model
    trained_model = train_model(
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        max_grad_norm=args.max_grad_norm,
        qdrant_client=qdrant_client,
        qdrant_collection=args.qdrant_collection,
        num_items=num_items,
        use_two_tower=args.use_two_tower,
        user_embedding_system=user_embedding_system
    )
    
    # Example of how to use the trained model and improvements for inference
    logger.info("Example search using trained model...")
    if search_engine is not None:
        # Example search query
        query = "mountain landscape"
        search_results = search_engine.search(
            query=query,
            search_type=SearchType(args.search_type),
            limit=10
        )
        
        # Log results
        logger.info(f"Search results for '{query}':")
        for i, result in enumerate(search_results[:3]):  # Show top 3 results
            logger.info(f"{i+1}. ID: {result.id}, Score: {result.score:.4f}")
    
    logger.info("Recommendation training and setup completed successfully.")

if __name__ == "__main__":
    main() 