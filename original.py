import argparse

import torch

import torch.nn as nn

import torch.optim as optim

import awswrangler as wr

from torch.utils.data import DataLoader

from qdrant_client import QdrantClient

# Define the UserTransformer Model (modified SASRec) for BPR

class SASRec(nn.Module):

def __init__(self, max_seq_len=50, embedding_dim=1024, num_layers=4, num_heads=8, dropout=0.2):

super(SASRec, self).__init__()

self.max_seq_len = max_seq_len

self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)

self.dropout = nn.Dropout(dropout)

encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True)

self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

def forward(self, x, mask):

batch_size, seq_len, _ = x.size()

positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

pos_emb = self.positional_embedding(positions)

x = x + pos_emb

x = self.dropout(x)

x = self.transformer_encoder(x, src_key_padding_mask=(mask == 0))

last_token_idx = mask.sum(dim=1).long() - 1

batch_indices = torch.arange(batch_size, device=x.device)

user_repr = x[batch_indices, last_token_idx, :]

return user_repr

# Define Dataset Class

class ReclrtableDataset(torch.utils.data.IterableDataset):

def __init__(self, train_parquet, val_parquet, chunk_size=10000, qdrant_url="http://localhost", qdrant_collection="images", qdrant_api_key=None):

self.train_parquet = train_parquet

self.chunk_size = chunk_size

self.qdrant_url = qdrant_url

self.qdrant_collection = qdrant_collection

self.qdrant_api_key = qdrant_api_key

self.val_df = wr.s3.read_parquet(path=val_parquet).reset_index(drop=True)

self.val_map = dict(zip(self.val_df['user_idx'], self.val_df['val']))

def __iter__(self):

client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

train_iter = wr.s3.read_parquet(path=self.train_parquet, chunked=True)

chunk_count = 0

for chunk in train_iter:

chunk_count += 1

chunk = chunk.reset_index(drop=True)

for idx, row in chunk.iterrows():

if idx % 1000 == 0:

print(f"Processing row {idx} in chunk {chunk_count}")

yield row

# Training Function

def train_model(train_loader, model, optimizer, scaler, device, epochs, checkpoint_dir, log_dir, max_grad_norm, qdrant_client, qdrant_collection, num_items):

model.train()

for epoch in range(epochs):

for batch in train_loader:

optimizer.zero_grad()

output = model(batch['train_seq_padded'].to(device), batch['train_seq_mask'].to(device))

loss = torch.nn.functional.mse_loss(output, batch['user_idx'].to(device).float())

loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

optimizer.step()

return model

# Main Function

def main():

parser = argparse.ArgumentParser()

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

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

sample_chunk = wr.s3.read_parquet(path=args.train_parquet, chunked=True).__next__().reset_index(drop=True)

train_max = sample_chunk['train_seq_padded'].apply(lambda s: max(s) if len(s) > 0 else 0).max()

val_df = wr.s3.read_parquet(path=args.val_parquet)

val_max = val_df['val'].max()

num_items = int(max(train_max, val_max)) + 1

print(f"Computed num_items: {num_items}")

qdrant_client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key, timeout=20)

train_dataset = ReclrtableDataset(args.train_parquet, args.val_parquet, args.chunk_size, args.qdrant_url, args.qdrant_collection, args.qdrant_api_key)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

model = SASRec(max_seq_len=50, embedding_dim=1024, num_layers=4, num_heads=8, dropout=0.2)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.use_amp) else None

trained_model = train_model(

train_loader, model, optimizer, scaler, device,

epochs=args.epochs,

checkpoint_dir=args.checkpoint_dir,

log_dir=args.log_dir,

max_grad_norm=args.max_grad_norm,

qdrant_client=qdrant_client,

qdrant_collection=args.qdrant_collection,

num_items=num_items

)

if __name__ == "__main__":

main()