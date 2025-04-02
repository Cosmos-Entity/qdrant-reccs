import argparse
import torch
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import re
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import logging
from tqdm import tqdm
import time
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range
from qdrant_client.http.models import SearchRequest, SearchParams
from data_preprocessing import QueryTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Define search types
class SearchType(str, Enum):
    VECTOR = "vector"
    TEXT = "text"
    HYBRID = "hybrid"
    PERSONALIZED = "personalized"

@dataclass
class SearchResult:
    """
    Container for search results with metadata.
    """
    id: Union[str, int]
    score: float
    vector_score: Optional[float] = None
    text_score: Optional[float] = None
    metadata_score: Optional[float] = None
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "score": self.score,
            "vector_score": self.vector_score,
            "text_score": self.text_score,
            "metadata_score": self.metadata_score
        }
        
        if self.payload:
            result["payload"] = self.payload
            
        return result

class EmbeddingModel:
    """
    Base class for text/query embedding models.
    """
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text to vector."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        return np.stack([self.embed_text(text) for text in texts])

class SentenceTransformerEmbedding(EmbeddingModel):
    """
    Text embedding using SentenceTransformer models.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = None):
        super().__init__(model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
    
    def embed_text(self, text: str) -> np.ndarray:
        if not text:
            # Return zero vector with correct dimensions for empty text
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        # Use the model's batch encoding capability
        return self.model.encode(texts, convert_to_numpy=True)

class CLIPEmbedding(EmbeddingModel):
    """
    Text embedding using CLIP models.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        super().__init__(model_name, device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
    
    def embed_text(self, text: str) -> np.ndarray:
        if not text:
            # Zero vector with correct dimensions
            return np.zeros(self.model.text_projection.out_features)
        
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        
        # Normalize and convert to numpy
        normalized = torch.nn.functional.normalize(outputs, p=2, dim=1)
        return normalized.cpu().numpy().squeeze()

class CustomEmbedding(EmbeddingModel):
    """
    Custom text embedding using Hugging Face models.
    Allows for more flexibility in model selection.
    """
    def __init__(self, model_name: str, device: str = None, pooling: str = "mean"):
        super().__init__(model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.pooling = pooling  # Options: mean, cls, max
    
    def embed_text(self, text: str) -> np.ndarray:
        if not text:
            # Get hidden size from model config
            hidden_size = self.model.config.hidden_size
            return np.zeros(hidden_size)
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Apply pooling strategy
        if self.pooling == "cls":
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0]
        elif self.pooling == "mean":
            # Mean pooling - take mean of all token embeddings
            attention_mask = inputs["attention_mask"]
            embeddings = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
        elif self.pooling == "max":
            # Max pooling - take max of all token embeddings
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = torch.max(outputs.last_hidden_state * attention_mask + (1 - attention_mask) * -1e9, dim=1)[0]
        
        # Normalize and convert to numpy
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized.cpu().numpy().squeeze()

class TextProcessor:
    """
    Handles text preprocessing for improved text search.
    """
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True, language: str = "english"):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.language = language
        
        # Initialize stemmer if needed
        self.stemmer = PorterStemmer() if use_stemming else None
        
        # Initialize stopwords if needed
        self.stop_words = set(stopwords.words(language)) if remove_stopwords else set()
    
    def preprocess(self, text: str) -> List[str]:
        """Preprocess text into tokens."""
        if not text:
            return []
        
        # Lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords if needed
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming if needed
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """Preprocess a batch of texts."""
        return [self.preprocess(text) for text in texts]

class BM25Scorer:
    """
    BM25 text scoring for keyword search.
    """
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
    
    def index_documents(self, documents: List[Dict[str, Any]], text_field: str = "description"):
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of documents with text content
            text_field: Field name containing text to index
        """
        # Extract text and preprocess
        self.documents = []
        self.doc_ids = []
        
        for doc in documents:
            doc_id = doc.get("id")
            text = doc.get("payload", {}).get(text_field, "")
            
            if text:
                self.documents.append(text)
                self.doc_ids.append(doc_id)
        
        # Preprocess documents
        tokenized_docs = self.text_processor.preprocess_batch(self.documents)
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[Union[str, int], float]]:
        """
        Search for documents matching query.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (doc_id, score) tuples
        """
        if not self.bm25:
            raise RuntimeError("Documents must be indexed before searching")
        
        # Preprocess query
        query_tokens = self.text_processor.preprocess(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with non-zero scores
                results.append((self.doc_ids[idx], scores[idx]))
        
        return results

class MetadataFilter:
    """
    Filter search results based on metadata.
    """
    def __init__(self):
        pass
    
    @staticmethod
    def create_filter(
        must: Optional[List[Dict]] = None,
        must_not: Optional[List[Dict]] = None,
        should: Optional[List[Dict]] = None,
        min_should_match: int = 1
    ) -> Dict:
        """
        Create a structured filter for Qdrant.
        
        Args:
            must: Conditions that must be satisfied
            must_not: Conditions that must not be satisfied
            should: Optional conditions where at least min_should_match should match
            min_should_match: Minimum number of should conditions to match
        
        Returns:
            Filter dictionary
        """
        filter_dict = {}
        
        if must:
            filter_dict["must"] = must
        
        if must_not:
            filter_dict["must_not"] = must_not
        
        if should:
            filter_dict["should"] = should
            filter_dict["min_should_match"] = min_should_match
        
        return filter_dict
    
    @staticmethod
    def exact_match_filter(field: str, value: Any) -> Dict:
        """Create exact match filter for a field."""
        return {field: {"$eq": value}}
    
    @staticmethod
    def range_filter(field: str, gt: Optional[float] = None, gte: Optional[float] = None, 
                    lt: Optional[float] = None, lte: Optional[float] = None) -> Dict:
        """Create range filter for numeric fields."""
        range_dict = {}
        
        if gt is not None:
            range_dict["$gt"] = gt
        if gte is not None:
            range_dict["$gte"] = gte
        if lt is not None:
            range_dict["$lt"] = lt
        if lte is not None:
            range_dict["$lte"] = lte
        
        return {field: range_dict}
    
    @staticmethod
    def text_match_filter(field: str, text: str, match_type: str = "exact") -> Dict:
        """
        Create text match filter.
        
        Args:
            field: Field to match on
            text: Text to match
            match_type: Type of match - 'exact', 'prefix', or 'contains'
        """
        if match_type == "exact":
            return {field: {"$eq": text}}
        elif match_type == "prefix":
            return {field: {"$like": f"{text}%"}}
        elif match_type == "contains":
            return {field: {"$like": f"%{text}%"}}
        else:
            raise ValueError(f"Unknown match_type: {match_type}")
    
    @staticmethod
    def in_list_filter(field: str, values: List[Any]) -> Dict:
        """Create filter for values in a list."""
        return {field: {"$in": values}}

class TextPreprocessor:
    """
    Text preprocessing for advanced search functionality.
    """
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text for search:
        1. Lowercase
        2. Tokenize
        3. Remove punctuation and special characters
        4. Remove stopwords
        5. Stemming
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]
        
        return tokens

class DiversityReranker:
    """
    Reranker that promotes diversity in search results.
    """
    def __init__(self, embedding_dim: int = 512, diversity_weight: float = 0.3):
        self.embedding_dim = embedding_dim
        self.diversity_weight = diversity_weight
    
    def rerank(self, item_ids: List[str], item_scores: List[float], 
              item_embeddings: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
        """
        Rerank items to promote diversity while maintaining relevance.
        
        Args:
            item_ids: List of item IDs
            item_scores: Original relevance scores for items
            item_embeddings: Dictionary mapping item IDs to embeddings
            
        Returns:
            Reranked list of (item_id, score) tuples
        """
        if not item_ids or len(item_ids) <= 1:
            return list(zip(item_ids, item_scores))
        
        # Normalize scores to [0, 1]
        max_score = max(item_scores)
        min_score = min(item_scores)
        score_range = max_score - min_score
        
        if score_range > 0:
            normalized_scores = [(score - min_score) / score_range for score in item_scores]
        else:
            normalized_scores = [1.0 for _ in item_scores]
        
        # Initialize reranked list with the highest scoring item
        best_idx = normalized_scores.index(max(normalized_scores))
        reranked_idxs = [best_idx]
        remaining_idxs = list(range(len(item_ids)))
        remaining_idxs.remove(best_idx)
        
        # Create embeddings array for efficient computation
        embeddings = np.array([item_embeddings[item_id] for item_id in item_ids])
        
        # Iteratively select items
        while remaining_idxs:
            # Calculate maximum similarity to already selected items
            similarities = np.zeros(len(remaining_idxs))
            
            for i, idx in enumerate(remaining_idxs):
                sim = 0
                for selected_idx in reranked_idxs:
                    # Cosine similarity
                    sim_to_selected = np.dot(embeddings[idx], embeddings[selected_idx]) / (
                        np.linalg.norm(embeddings[idx]) * np.linalg.norm(embeddings[selected_idx]) + 1e-8
                    )
                    sim = max(sim, sim_to_selected)
                similarities[i] = sim
            
            # Calculate diversity scores (1 - max_similarity)
            diversity_scores = 1 - similarities
            
            # Calculate combined scores
            relevance_weight = 1 - self.diversity_weight
            combined_scores = [
                relevance_weight * normalized_scores[idx] + self.diversity_weight * div_score
                for idx, div_score in zip(remaining_idxs, diversity_scores)
            ]
            
            # Select the best item
            best_remaining_idx = remaining_idxs[combined_scores.index(max(combined_scores))]
            reranked_idxs.append(best_remaining_idx)
            remaining_idxs.remove(best_remaining_idx)
        
        # Create reranked result
        reranked_results = [
            (item_ids[idx], item_scores[idx])
            for idx in reranked_idxs
        ]
        
        return reranked_results

class ContentTypePreference:
    """
    Manages content type preferences for steering search results.
    """
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.preference_vectors = {}
        
    def add_preference_vector(self, name: str, vector: np.ndarray):
        """
        Add a content preference vector.
        
        Args:
            name: Name of the preference
            vector: Embedding vector representing the preference
        """
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        self.preference_vectors[name] = vector
    
    def steer_results(self, results: List[Dict[str, Any]], 
                     preference_name: str, 
                     weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Steer search results toward a specific content preference.
        
        Args:
            results: Original search results
            preference_name: Name of the preference to steer toward
            weight: How heavily to weight the preference (0-1)
            
        Returns:
            Reordered search results
        """
        if preference_name not in self.preference_vectors:
            return results
        
        preference_vector = self.preference_vectors[preference_name]
        
        # Calculate similarity to preference vector
        for result in results:
            embedding = np.array(result.get('payload', {}).get('embedding', []))
            if len(embedding) == 0:
                result['preference_score'] = 0
                continue
                
            # Normalize embedding if needed
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Calculate cosine similarity to preference vector
            similarity = np.dot(embedding, preference_vector)
            result['preference_score'] = similarity
        
        # Combine original scores with preference scores
        for result in results:
            original_score = result.get('score', 0)
            preference_score = result.get('preference_score', 0)
            result['combined_score'] = (1 - weight) * original_score + weight * preference_score
        
        # Sort by combined score
        results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return results

class AdvancedSearch:
    """
    Advanced search system that combines vector similarity with text relevance
    and diversity reranking.
    """
    def __init__(self, qdrant_client: QdrantClient, collection_name: str,
                use_query_transformation: bool = True, 
                embedding_dim: int = 512, 
                diversity_weight: float = 0.3):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Initialize components
        self.query_transformer = QueryTransformer() if use_query_transformation else None
        self.text_preprocessor = TextPreprocessor()
        self.diversity_reranker = DiversityReranker(
            embedding_dim=embedding_dim,
            diversity_weight=diversity_weight
        )
        self.content_preferences = ContentTypePreference(embedding_dim=embedding_dim)
        
        # Initialize BM25 scorer (will need to be built)
        self.bm25_scorer = BM25Scorer(self.text_preprocessor)
        self.bm25_index_built = False
    
    def build_text_index(self):
        """
        Build the text search index for hybrid search.
        """
        # Get all items from the collection with their text descriptions
        items = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=10000  # Adjust based on collection size
        )[0]
        
        # Extract document texts for BM25 indexing
        documents = {}
        for i, item in enumerate(items):
            item_id = item.id
            
            # Extract text from payload
            description = item.payload.get('description', '')
            title = item.payload.get('title', '')
            
            # Combine text fields
            text = f"{title} {description}"
            
            documents[item_id] = text
        
        # Build the BM25 index
        self.bm25_scorer.index_documents(documents)
        self.bm25_index_built = True
    
    def search(self, query: str, 
              filter_conditions: Optional[Dict[str, Any]] = None,
              limit: int = 20,
              preference: Optional[str] = None,
              preference_weight: float = 0.5,
              use_text_search: bool = True,
              use_vector_search: bool = True,
              use_diversity: bool = True,
              vector_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform advanced hybrid search.
        
        Args:
            query: Search query
            filter_conditions: Optional filtering conditions
            limit: Maximum number of results
            preference: Optional content preference to steer toward
            preference_weight: Weight of the preference steering
            use_text_search: Whether to use text search
            use_vector_search: Whether to use vector search
            use_diversity: Whether to use diversity reranking
            vector_weight: Weight of vector search vs text search
            
        Returns:
            Search results
        """
        # Transform query if enabled
        if self.query_transformer is not None:
            transformed_query = self.query_transformer.transform_query(query)
        else:
            transformed_query = query
        
        # Convert filter conditions to Qdrant format
        filter_obj = None
        if filter_conditions:
            conditions = []
            for field, value in filter_conditions.items():
                if isinstance(value, dict) and ('min' in value or 'max' in value):
                    # Range filter
                    range_params = {}
                    if 'min' in value:
                        range_params['gte'] = value['min']
                    if 'max' in value:
                        range_params['lte'] = value['max']
                    conditions.append(FieldCondition(key=field, range=Range(**range_params)))
                else:
                    # Match filter
                    conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
            
            filter_obj = Filter(must=conditions)
        
        results = []
        
        # Vector search
        if use_vector_search:
            # This assumes the query can be directly transformed to a vector
            # In a real implementation, you would use a model to convert the query to a vector
            vector_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=None,  # Replace with actual vector from query
                query_filter=filter_obj,
                limit=limit * 2,  # Get more results for merging/reranking
                with_payload=True,
                with_vectors=True
            )
            
            results.extend(vector_results)
        
        # Text search using BM25
        if use_text_search and self.bm25_index_built:
            # Get BM25 scores
            bm25_scores = self.bm25_scorer.search(transformed_query)
            
            # Get items by IDs with scores
            text_result_ids = list(bm25_scores.keys())
            
            if text_result_ids:
                # Get items from Qdrant
                text_items = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=text_result_ids,
                    with_payload=True,
                    with_vectors=True
                )
                
                # Add BM25 scores
                for item in text_items:
                    item_dict = item.dict()
                    item_dict['score'] = bm25_scores.get(item.id, 0)
                    results.append(item_dict)
        
        # Merge results if both search types were used
        if use_text_search and use_vector_search:
            # Normalize vector search scores
            if results:
                max_vector_score = max([r.get('score', 0) for r in results if 'score' in r], default=1)
                for r in results:
                    if 'score' in r:
                        r['vector_score'] = r['score'] / max_vector_score if max_vector_score > 0 else 0
            
            # Normalize BM25 scores
            if results:
                max_bm25_score = max([r.get('bm25_score', 0) for r in results if 'bm25_score' in r], default=1)
                for r in results:
                    if 'bm25_score' in r:
                        r['text_score'] = r['bm25_score'] / max_bm25_score if max_bm25_score > 0 else 0
            
            # Combine scores
            for r in results:
                vector_score = r.get('vector_score', 0)
                text_score = r.get('text_score', 0)
                r['score'] = vector_weight * vector_score + (1 - vector_weight) * text_score
            
            # Remove duplicate items (keep the highest scored one)
            seen_ids = {}
            unique_results = []
            for r in results:
                item_id = r.get('id')
                if item_id not in seen_ids or r.get('score', 0) > seen_ids[item_id]:
                    seen_ids[item_id] = r.get('score', 0)
                    unique_results.append(r)
            results = unique_results
        
        # Sort by score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Apply content preference steering if requested
        if preference is not None and preference in self.content_preferences.preference_vectors:
            results = self.content_preferences.steer_results(
                results, preference, preference_weight
            )
        
        # Apply diversity reranking if requested
        if use_diversity and len(results) > 1:
            # Extract item IDs, scores, and embeddings
            item_ids = [r.get('id') for r in results]
            item_scores = [r.get('score', 0) for r in results]
            item_embeddings = {
                r.get('id'): np.array(r.get('vector', []))
                for r in results
                if 'vector' in r and r.get('id') is not None
            }
            
            # Only rerank items with embeddings
            rerank_ids = [item_id for item_id in item_ids if item_id in item_embeddings]
            rerank_scores = [score for item_id, score in zip(item_ids, item_scores) if item_id in item_embeddings]
            
            if rerank_ids:
                reranked = self.diversity_reranker.rerank(rerank_ids, rerank_scores, item_embeddings)
                
                # Create a lookup for original results
                result_lookup = {r.get('id'): r for r in results}
                
                # Build reranked results
                final_results = [result_lookup[item_id] for item_id, _ in reranked]
                
                # Add any results without embeddings at the end
                for item_id in item_ids:
                    if item_id not in item_embeddings and item_id in result_lookup:
                        final_results.append(result_lookup[item_id])
                
                results = final_results[:limit]
            
        # Truncate to requested limit
        return results[:limit]
    
    def add_content_preference(self, name: str, examples: List[str]):
        """
        Add a content preference based on example items.
        
        Args:
            name: Name of the preference
            examples: List of example item IDs representing the preference
        """
        # Get item vectors
        items = self.client.retrieve(
            collection_name=self.collection_name,
            ids=examples,
            with_vectors=True
        )
        
        if not items:
            return False
        
        # Average the vectors to create a preference vector
        vectors = [np.array(item.vector) for item in items if hasattr(item, 'vector') and item.vector]
        
        if not vectors:
            return False
        
        preference_vector = np.mean(vectors, axis=0)
        
        # Add to preferences
        self.content_preferences.add_preference_vector(name, preference_vector)
        return True

def main():
    parser = argparse.ArgumentParser(description="Advanced search for vector databases")
    
    parser.add_argument("--query", type=str, required=True,
                      help="Search query")
    
    parser.add_argument("--search-type", type=str, choices=["vector", "text", "hybrid", "personalized"], default="hybrid",
                      help="Type of search to perform")
    
    parser.add_argument("--qdrant-url", type=str, default="http://localhost",
                      help="URL of Qdrant service")
    
    parser.add_argument("--qdrant-api-key", type=str, default=None,
                      help="API key for Qdrant service")
    
    parser.add_argument("--collection", type=str, default="elements",
                      help="Name of collection to search")
    
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                      help="Embedding model to use for query")
    
    parser.add_argument("--limit", type=int, default=10,
                      help="Number of results to return")
    
    parser.add_argument("--user-id", type=str, default=None,
                      help="User ID for personalized search")
    
    parser.add_argument("--vector-weight", type=float, default=0.7,
                      help="Weight for vector search results")
    
    parser.add_argument("--text-weight", type=float, default=0.3,
                      help="Weight for text search results")
    
    parser.add_argument("--diversity", type=float, default=0.0,
                      help="Diversity factor (0-1)")
    
    parser.add_argument("--reranking-model", type=str, default=None,
                      help="Model to use for reranking")
    
    args = parser.parse_args()
    
    # Initialize embedding model
    if "clip" in args.embedding_model.lower():
        embedding_model = CLIPEmbedding(model_name=args.embedding_model)
    else:
        embedding_model = SentenceTransformerEmbedding(model_name=args.embedding_model)
    
    # Initialize search engine
    search_engine = AdvancedSearch(
        qdrant_client=QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key),
        collection_name=args.collection,
        embedding_dim=embedding_model.model.get_sentence_embedding_dimension()
    )
    
    # Perform search
    results = search_engine.search(
        query=args.query,
        search_type=SearchType(args.search_type),
        limit=args.limit,
        user_id=args.user_id,
        diversity_factor=args.diversity,
        reranking_model=args.reranking_model
    )
    
    # Print results
    print(f"Search results for query: '{args.query}'")
    print("-" * 50)
    
    for i, result in enumerate(results):
        print(f"{i+1}. ID: {result.id}, Score: {result.score:.4f}")
        if result.vector_score:
            print(f"   Vector Score: {result.vector_score:.4f}")
        if result.text_score:
            print(f"   Text Score: {result.text_score:.4f}")
        
        description = result.payload.get("description", "")
        if description:
            print(f"   Description: {description[:100]}...")
        
        print()

if __name__ == "__main__":
    main() 