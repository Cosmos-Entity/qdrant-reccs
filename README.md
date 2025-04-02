# Advanced Recommendation System

## Project Structure

### Core Recommendation System Files
- `data_preprocessing.py` - Query transformation implementation
- `three_vector_embedding.py` - Three-vector architecture implementation
- `user_embeddings.py` - User embeddings in latent space implementation
- `two_tower_model.py` - Two-tower model implementation 
- `advanced_search.py` - Advanced search implementation
- `improved_recommender.py` - Main recommendation system implementation

### Auxiliary Files
- `reccs/` - Main package directory with models, utils, and tests
  - `models/` - Neural network models for recommendation
  - `utils/` - Utility functions for recommendation system
  - `tests/` - Test cases for recommendation components
  - `data/` - Data processing utilities

### Snapshot and Debugging Files
These files are not part of the core recommendation system and are used for snapshot management and debugging:
- `upload_snapshots.py` - Uploads snapshots to Qdrant 
- `extract_snapshot_data.py` - Extracts data from Qdrant snapshots
- `fix_snapshot_config.py` - Fixes snapshot configuration issues
- `fix_snapshot_config_backup.py` - Backup of snapshot configuration fix
- `restore_snapshots.py` - Restores collections from snapshots
- `check_collections.py` - Verifies collection status
- `examine_element_payload.py` - Debug tool for examining element data
- `examine_user_payload.py` - Debug tool for examining user data
- `examine_users.py` - Debug tool for user data analysis
- `test_qdrant.py` - Simple Qdrant connection test

## Key Improvements and Benefits

This recommendation system implements five major improvements that address common challenges in content recommendation:

### 1. Query Transformation
- **Challenge:** Basic search terms can return inappropriate or irrelevant content
- **Solution:** Dual-approach system combining rule-based replacement and neural transformation
- **Advantage:** Preserves search intent while ensuring appropriate results through intelligent query rewriting

### 2. Three-Vector Architecture
- **Challenge:** Single CLIP embeddings don't capture all relevant content signals
- **Solution:** Combined representation using image, text, and metadata vectors with adaptive weighting
- **Advantage:** Richer content representation that captures multiple aspects of items

### 3. User Embeddings in Latent Space
- **Challenge:** Users and content exist in separate representation spaces
- **Solution:** Neural transformation of user interactions into CLIP-compatible embeddings
- **Advantage:** Enables direct similarity computation between users and content

### 4. Two-Tower Model
- **Challenge:** Independent learning of user and item representations
- **Solution:** Contrastive learning approach with separate user and item towers
- **Advantage:** Joint optimization of user and item representations for better matching

### 5. Advanced Search
- **Challenge:** Basic vector similarity search has limited capabilities
- **Solution:** Hybrid search combining vector similarity, text relevance, and diversity reranking
- **Advantage:** More comprehensive search that balances multiple relevance signals

## Overview

This project enhances a recommendation system with five major improvements that address key challenges in content recommendation:

1. **Query Transformation**
2. **Three-Vector Architecture**
3. **User Embeddings in Latent Space**
4. **Two-Tower Model**
5. **Advanced Search**

The system is designed to provide more relevant, diverse, and appropriate recommendations while placing users and content in the same latent space for direct similarity computation.

## Technical Implementation

### 1. Query Transformation (`data_preprocessing.py`)

**Problem:** Basic search terms can produce unwanted content (e.g., "girl" returning sensual images).

**Solution:** A dual-approach system that transforms problematic queries:
- **Rule-based replacement:** Dictionary of problematic terms with safer alternatives
- **Neural transformation:** Sequence-to-sequence model that rewrites queries while preserving core intent

```python
transformed_query = query_transformer.transform_query("girl with a camera")
# Output: "woman with a camera"
```

**Advantages over alternatives:**
- More flexible than pure rule-based systems
- More efficient than post-filtering
- Better preserves search intent than query blocking

### 2. Three-Vector Architecture (`three_vector_embedding.py`)

**Problem:** CLIP embeddings alone don't capture all relevant signals, and metadata is unused.

**Solution:** A unified representation system with three vector types:
- **Image Vector:** CLIP embedding of the visual content
- **Text Description Vector:** Generated from LLM captioning
- **Metadata Vector:** Structured attributes encoded separately

```python
combined_vector = three_vector_encoder(
    image=image_tensor,
    text_description={"text_ids": text_ids, "attention_mask": mask},
    metadata={"categorical_category": category_id, "text_tags": tags}
)
```

**Dynamic weighting:** Neural attention mechanism that adaptively weights vectors based on query context.

**Advantages over alternatives:**
- Richer representation than single-vector approaches
- More adaptable than fixed-weight combinations
- Preserves original signal fidelity better than early fusion

### 3. User Embeddings in Latent Space (`user_embeddings.py`)

**Problem:** Users not represented in the same space as content, making direct similarity impossible.

**Solution:** Neural transformation of user interaction history into CLIP-compatible embeddings:
- **Sequential modeling:** Transformer that encodes interaction sequences
- **CLIP projection:** Maps user representations to content space
- **Adaptive interest mapping:** Captures multiple facets of user preferences

```python
user_embedding = user_embedding_model(
    item_embeddings=history_sequence,
    attention_mask=sequence_mask
)
# Result: User embedding in the same space as content
```

**Advantages over alternatives:**
- More expressive than matrix factorization
- Directly comparable to content embeddings
- Self-updating as users interact with content

### 4. Two-Tower Model (`two_tower_model.py`)

**Problem:** No joint learning of user and item representations.

**Solution:** Contrastive learning approach with:
- **User tower:** Processes interaction histories
- **Item tower:** Processes content features
- **Contrastive objective:** Brings positive user-item pairs closer together

```python
loss = two_tower_model.compute_contrastive_loss(
    user_embeddings=user_emb,
    positive_items=pos_items,
    negative_items=neg_items
)
```

**Advantages over alternatives:**
- More efficient inference than cross-attention
- Better quality than independent encoders
- Enables fast retrieval at scale

### 5. Advanced Search (`advanced_search.py`)

**Problem:** Limited search capabilities that miss relevant content.

**Solution:** Hybrid search combining:
- **Vector similarity:** For semantic understanding
- **Text relevance:** BM25 scoring for lexical matching
- **Diversity reranking:** Promotes variety in results
- **Content steering:** Adjustable preference vectors

```python
results = advanced_search.search(
    query="mountain landscape",
    preference="artistic",
    preference_weight=0.7,
    use_diversity=True
)
```

**Advantages over alternatives:**
- Balances semantic and lexical matching
- Produces more diverse result sets
- Allows steering without retraining

## Implementation Details

### Query Construction

The combined vector generation process:

```
1. Query → QueryTransformer → Transformed Query
2. Transformed Query → ThreeVectorEncoder → Query Vector
3. Query Vector + Filters → AdvancedSearch → Results
```

### Vector Updates

The system implements incremental vector updates:

```
1. New Content → ThreeVectorEncoder → Content Vector
2. Content Vector → VectorStorage.add() → Indexed Vector
3. User Interactions → UserEmbeddingModel → Updated User Vector
```

### Self-Running System

The recommendation system is designed to be self-running with minimal tuning:

- Automatic query transformation based on problematic term detection
- Learned vector weights adapted to query context
- Continuous user embedding updates based on interactions
- Diversity parameters self-adjusting based on result analysis

## Getting Started

### Prerequisites

```
torch>=1.9.0
transformers>=4.11.0
qdrant-client>=0.11.0
nltk>=3.6.0
numpy>=1.20.0
```

### Basic Usage

```python
# Initialize components
query_transformer = QueryTransformer()
three_vector_encoder = ThreeVectorEncoder()
user_embedding_model = UserEmbeddingModel()
advanced_search = AdvancedSearch(qdrant_client, "content_collection")

# Transform query
safe_query = query_transformer.transform_query("girl with camera")

# Generate query vector
query_vector = three_vector_encoder.encode_query(
    query_text={"text_ids": tokenize(safe_query)}
)

# Get user embedding
user_vector = user_embedding_model.compute_user_embedding(user_history)

# Search with combined signals
results = advanced_search.search(
    query=safe_query,
    query_vector=query_vector,
    user_vector=user_vector,
    limit=20
)
```

## Performance Considerations

- Vector quantization reduces storage requirements
- Batched processing for neural transformations improves throughput
- Caching frequently accessed embeddings reduces latency
- Partitioned vector indices enable efficient filtering

## Future Improvements

- Pre-computed cluster centroids for faster initial retrieval
- Multi-stage ranking with lightweight first-pass filters
- Cross-modal alignment fine-tuning
- Federated user representation learning 