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
from utils.image_utils import url_to_base64
import os
import json
from sklearn.preprocessing import normalize
import datetime
import requests
from io import BytesIO

class LLMCaptionGenerator:
    """
    Generate captions for images using LLM.
    """
    def __init__(self, api_key=None, model="o1"):
        if api_key:
            openai.api_key = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            openai.api_key = os.environ.get("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided or set as environment variable")
            
        self.model = model
        self.prompt_template = """
        You are an unflinchingly honest reviewer who adapts to different types of images and art forms. Begin by determining whether the image is:
        1.	Serious Artwork or Photograph (e.g., fine art, editorial, design, conceptual):
            •	Detailed Critical Analysis (concisely, in 3-5 sentences) focusing on when appropriate:
                •	Main Subject & Visual Content: Precisely describe the subjects and notable elements.
                •	Style & Composition: Use advanced artistic or photographic terminology (e.g., "chiaroscuro," "Dutch angle," "golden ratio," "bokeh," "postmodern," "Baroque influence") where relevant. Mention any recognizable time period, movement, or technique (e.g., Renaissance, Pop Art, Minimalist). If you suspect a certain camera type or lens, feel free to note it.
                •	Colors & Aesthetics: The color palette's harmony or discord, calling out clichés or praising truly innovative choices.
                •	Emotional Impact: The intended mood, noting if it's truly evocative or falls flat.
                •	Technical Quality: Assess focus, lighting, post-processing, and any advanced or flawed techniques.
            •	Ratings (1–10):
                •	Composition
                •	Technical Quality
                •	Artistic Value (skew toward originality, depth, and sophistication)
                •	Originality (penalize clichés and "cheuggy" trends; reward fresh ideas)
                •	Format: In 1-2 sentences, provide an honest, well-reasoned critique (with genuine praise only where earned). Avoid empty flattery, but do not hesitate to acknowledge a truly exceptional piece.
        2.	Casual Photo, Meme, Selfie, or Otherwise Non-Art
            •	Brief Description: Summarize what's happening or who is pictured (if clear). Note if it's funny, dull, mundane, cringe, etc.
            •	Concise Feedback: No need for deep composition or color theory critiques. Only mention basics if relevant (e.g., blatantly poor lighting or focus).

        Overall Guidelines:
            •	Stay unfiltered, direct, and avoid hollow niceties.
            •	Provide technical or historical insight where appropriate—if the style suggests a known era, method, or movement, mention it.
            •	Critique fairly: highlight strengths if they truly exist, yet be brutally honest about mediocrity, clichés, or overdone aesthetics.
                    """
    
    def  generate_caption(self, image_url: str) -> str:
        """
        Generate a caption for the image using the LLM.
        """
        # Using the new OpenAI API format (1.0.0+)
        client = openai.OpenAI(api_key=openai.api_key)

        # Ensure image URL ends with ?format=jpeg
        if not image_url.endswith("?format=jpeg"):
            image_url = f"{image_url}?format=jpeg"

        base64_url = url_to_base64(image_url)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt_template},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": base64_url}}
                ]}
            ]
        )
        return response.choices[0].message.content

class TextEmbeddingModel:
    """
    Generate text embeddings using a model other than CLIP.
    """
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        from transformers import AutoTokenizer, AutoModel
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        # Lazy loading - will load when first needed
        
    def _ensure_model_loaded(self):
        """
        Ensure the model and tokenizer are loaded.
        Implements lazy loading to save memory until needed.
        """
        if self.model is None or self.tokenizer is None:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()  # Set to evaluation mode
    
    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Generate an embedding for the text using BGE model.
        
        Args:
            text: Input text to embed
            
        Returns:
            Text embedding tensor
        """
        return self.get_embeddings_batch([text]).squeeze()
    
    def get_embeddings_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            Batch of text embedding tensors
        """
        self._ensure_model_loaded()
        
        # Prepare inputs
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # For BGE models, use the [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0]
            
            # Normalize embeddings (important for BGE models)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings


class MetadataProcessor:
    """
    Process metadata into a format optimized for text embedding models.
    Structured for BGE and similar embedding models.
    """
    def __init__(self):
        self.max_tags = 50
        self.max_colors = 10
    
    def process_metadata(self, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Convert metadata dictionary to a structured string for embedding and a clean dictionary.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Tuple of (formatted_string, processed_metadata_dict)
        """
        parts = []
        processed_dict = {}
        
        # ===== Basic Content Type and Source =====
        # These fields define what the item fundamentally is
        if metadata.get("contentType"):
            parts.append(f"contentType={metadata['contentType']}")
            processed_dict["contentType"] = metadata["contentType"]
        
        if metadata.get("type"):
            parts.append(f"type={metadata['type']}")
            processed_dict["type"] = metadata["type"]
        
        if metadata.get("createdAt"):
            parts.append(f"createdAt={metadata['createdAt']}")
            processed_dict["createdAt"] = metadata["createdAt"]
        
        # ===== Visual Content Tags and Colors =====
        # Computer vision tags (critical for visual search)
        if metadata.get("tags"):
            tags = metadata["tags"][:self.max_tags]
            parts.append(f"tags=[{' '.join(tags)}]")
            processed_dict["tags"] = tags
        
        # Dominant colors
        if metadata.get("colors"):
            colors = metadata["colors"][:self.max_colors]
            parts.append(f"colors=[{' '.join(colors)}]")
            processed_dict["colors"] = colors
        
        # ===== User-Added Tags =====
        if metadata.get("userTags"):
            all_user_tags = []
            for tag_item in metadata["userTags"]:
                if "tags" in tag_item and tag_item["tags"]:
                    all_user_tags.extend(tag_item["tags"])
            
            if all_user_tags:
                unique_user_tags = list(set(all_user_tags))[:self.max_tags]
                parts.append(f"userTags=[{' '.join(unique_user_tags)}]")
                processed_dict["userTags"] = unique_user_tags
        
        # ===== Source Platform Information =====
        
        # Instagram
        if metadata.get("instagramAuthorFullName"):
            parts.append(f"instagramAuthorFullName={metadata['instagramAuthorFullName']}")
            processed_dict["instagramAuthorFullName"] = metadata["instagramAuthorFullName"]
        
        if metadata.get("instagramAuthorName"):
            parts.append(f"instagramAuthorName={metadata['instagramAuthorName']}")
            processed_dict["instagramAuthorName"] = metadata["instagramAuthorName"]
        
        if metadata.get("instagramCaption"):
            caption = metadata["instagramCaption"].replace('\n', ' ')
            parts.append(f"instagramCaption={caption}")
            processed_dict["instagramCaption"] = caption
        
        if metadata.get("instagramContentAccessibility"):
            parts.append(f"instagramContentAccessibility={metadata['instagramContentAccessibility']}")
            processed_dict["instagramContentAccessibility"] = metadata["instagramContentAccessibility"]
        
        if "isInstagramCarousel" in metadata:
            parts.append(f"isInstagramCarousel={metadata['isInstagramCarousel']}")
            processed_dict["isInstagramCarousel"] = metadata["isInstagramCarousel"]
        
        # YouTube
        if metadata.get("youtubeTitle"):
            parts.append(f"youtubeTitle={metadata['youtubeTitle']}")
            processed_dict["youtubeTitle"] = metadata["youtubeTitle"]
        
        if metadata.get("youtubeAuthorName"):
            parts.append(f"youtubeAuthorName={metadata['youtubeAuthorName']}")
            processed_dict["youtubeAuthorName"] = metadata["youtubeAuthorName"]
        
        # TikTok
        if metadata.get("tikTokAuthorName"):
            parts.append(f"tikTokAuthorName={metadata['tikTokAuthorName']}")
            processed_dict["tikTokAuthorName"] = metadata["tikTokAuthorName"]
        
        # Twitter
        if metadata.get("twitterAuthorName"):
            parts.append(f"twitterAuthorName={metadata['twitterAuthorName']}")
            processed_dict["twitterAuthorName"] = metadata["twitterAuthorName"]
        
        if metadata.get("twitterAuthorUsername"):
            parts.append(f"twitterAuthorUsername={metadata['twitterAuthorUsername']}")
            processed_dict["twitterAuthorUsername"] = metadata["twitterAuthorUsername"]
        
        if "isTwitterCarousel" in metadata:
            parts.append(f"isTwitterCarousel={metadata['isTwitterCarousel']}")
            processed_dict["isTwitterCarousel"] = metadata["isTwitterCarousel"]
        
        # Pinterest
        if metadata.get("pinterestAuthorName"):
            parts.append(f"pinterestAuthorName={metadata['pinterestAuthorName']}")
            processed_dict["pinterestAuthorName"] = metadata["pinterestAuthorName"]
        
        if metadata.get("pinterestAuthorUsername"):
            parts.append(f"pinterestAuthorUsername={metadata['pinterestAuthorUsername']}")
            processed_dict["pinterestAuthorUsername"] = metadata["pinterestAuthorUsername"]
        
        # URLs
        if metadata.get("url"):
            parts.append(f"url={metadata['url']}")
            processed_dict["url"] = metadata["url"]
        
        if metadata.get("sourceUrl") and metadata.get("sourceUrl") != metadata.get("url"):
            parts.append(f"sourceUrl={metadata['sourceUrl']}")
            processed_dict["sourceUrl"] = metadata["sourceUrl"]
        
        # ===== Media-Specific Content =====
        
        # Product information
        if metadata.get("productBrand"):
            parts.append(f"productBrand={metadata['productBrand']}")
            processed_dict["productBrand"] = metadata["productBrand"]
        
        if metadata.get("productTitle"):
            parts.append(f"productTitle={metadata['productTitle']}")
            processed_dict["productTitle"] = metadata["productTitle"]
        
        if metadata.get("productDescription"):
            desc = metadata["productDescription"].replace('\n', ' ')
            parts.append(f"productDescription={desc}")
            processed_dict["productDescription"] = desc
        
        # Article information
        if metadata.get("articleTitle"):
            parts.append(f"articleTitle={metadata['articleTitle']}")
            processed_dict["articleTitle"] = metadata["articleTitle"]
        
        if metadata.get("articleDescription"):
            desc = metadata["articleDescription"].replace('\n', ' ')
            parts.append(f"articleDescription={desc}")
            processed_dict["articleDescription"] = desc
        
        # Movie information
        if metadata.get("movieTitle"):
            parts.append(f"movieTitle={metadata['movieTitle']}")
            processed_dict["movieTitle"] = metadata["movieTitle"]
        
        if metadata.get("movieYear"):
            parts.append(f"movieYear={metadata['movieYear']}")
            processed_dict["movieYear"] = metadata["movieYear"]
        
        # Text within image (OCR)
        if metadata.get("textWithinImage"):
            text = metadata["textWithinImage"].replace('\n', ' ')
            parts.append(f"textWithinImage={text}")
            processed_dict["textWithinImage"] = text
        
        # ===== Engagement & Relation Information =====
        if "isPrivate" in metadata:
            parts.append(f"isPrivate={metadata['isPrivate']}")
            processed_dict["isPrivate"] = metadata["isPrivate"]
        
        if "numberOfConnections" in metadata and metadata["numberOfConnections"] is not None:
            parts.append(f"numberOfConnections={metadata['numberOfConnections']}")
            processed_dict["numberOfConnections"] = metadata["numberOfConnections"]
        
        if metadata.get("publicUserIds"):
            parts.append(f"publicUserCount={len(metadata['publicUserIds'])}")
            processed_dict["publicUserCount"] = len(metadata['publicUserIds'])
            processed_dict["publicUserIds"] = metadata["publicUserIds"]
        
        if metadata.get("contextualSearchUserIds"):
            parts.append(f"contextualSearchUserCount={len(metadata['contextualSearchUserIds'])}")
            processed_dict["contextualSearchUserCount"] = len(metadata['contextualSearchUserIds'])
            processed_dict["contextualSearchUserIds"] = metadata["contextualSearchUserIds"]
        
        if metadata.get("categoryIds"):
            # Include count
            parts.append(f"categoryCount={len(metadata['categoryIds'])}")
            processed_dict["categoryCount"] = len(metadata['categoryIds'])
            
            # Include specific IDs if not too many
            if len(metadata["categoryIds"]) > 0 and len(metadata["categoryIds"]) <= 10:
                cat_ids = [str(cid) for cid in metadata["categoryIds"]]
                parts.append(f"categoryIds=[{' '.join(cat_ids)}]")
                processed_dict["categoryIds"] = metadata["categoryIds"]
        
        # ===== Cluster & Owner Information =====
        if metadata.get("originalCluster") and isinstance(metadata["originalCluster"], dict):
            cluster = metadata["originalCluster"]
            processed_dict["originalCluster"] = cluster
            
            if "id" in cluster:
                parts.append(f"originalClusterId={cluster['id']}")
                processed_dict["originalClusterId"] = cluster['id']
            
            if "isBanned" in cluster:
                parts.append(f"isClusterBanned={cluster['isBanned']}")
                processed_dict["isClusterBanned"] = cluster['isBanned']
            
            if "isAesthetic" in cluster and cluster["isAesthetic"] is not None:
                parts.append(f"isClusterAesthetic={cluster['isAesthetic']}")
                processed_dict["isClusterAesthetic"] = cluster["isAesthetic"]
        
        if metadata.get("ownerId"):
            parts.append(f"ownerId={metadata['ownerId']}")
            processed_dict["ownerId"] = metadata["ownerId"]
        
        if "isOwnerBanned" in metadata and metadata["isOwnerBanned"] is not None:
            parts.append(f"isOwnerBanned={metadata['isOwnerBanned']}")
            processed_dict["isOwnerBanned"] = metadata["isOwnerBanned"]
        
        # ===== Safety & Moderation Information =====
        if "isNotSafeForWork" in metadata:
            parts.append(f"isNotSafeForWork={metadata['isNotSafeForWork']}")
            processed_dict["isNotSafeForWork"] = metadata["isNotSafeForWork"]
        
        if metadata.get("notSafeForWorkStatus"):
            parts.append(f"notSafeForWorkStatus={metadata['notSafeForWorkStatus']}")
            processed_dict["notSafeForWorkStatus"] = metadata["notSafeForWorkStatus"]
        
        # Include all available scores
        if "aiGeneratedScore" in metadata and metadata["aiGeneratedScore"] is not None:
            parts.append(f"aiGeneratedScore={metadata['aiGeneratedScore']}")
            processed_dict["aiGeneratedScore"] = metadata["aiGeneratedScore"]
        
        if "explicitNsfwScore" in metadata and metadata["explicitNsfwScore"] is not None:
            parts.append(f"explicitNsfwScore={metadata['explicitNsfwScore']}")
            processed_dict["explicitNsfwScore"] = metadata["explicitNsfwScore"]
        
        if "suggestiveNsfwScore" in metadata and metadata["suggestiveNsfwScore"] is not None:
            parts.append(f"suggestiveNsfwScore={metadata['suggestiveNsfwScore']}")
            processed_dict["suggestiveNsfwScore"] = metadata["suggestiveNsfwScore"]
        
        # ===== Image Properties =====
        if metadata.get("image") and isinstance(metadata["image"], dict):
            img = metadata["image"]
            processed_dict["image"] = img
            
            if "width" in img and "height" in img:
                parts.append(f"imageDimensions={img['width']}x{img['height']}")
                processed_dict["imageDimensions"] = f"{img['width']}x{img['height']}"
                processed_dict["imageWidth"] = img['width']
                processed_dict["imageHeight"] = img['height']
            
            if "isAnimated" in img:
                parts.append(f"isAnimated={img['isAnimated']}")
                processed_dict["isAnimated"] = img['isAnimated']
            
            if "hash" in img:
                parts.append(f"imageHash={img['hash']}")
                processed_dict["imageHash"] = img['hash']
        
        # Join all parts with spaces for the string representation
        metadata_str = " ".join(parts)
        
        return metadata_str, processed_dict

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
            # Download and process the image from URL
            from PIL import Image
            from io import BytesIO
            
            # Ensure URL ends with format=jpeg for compatibility
            if not image_url.endswith("?format=jpeg"):
                image_url = f"{image_url}?format=jpeg"
            
            # Download the image
            response = requests.get(image_url, stream=True)
            if response.status_code != 200:
                raise ValueError(f"Failed to download image from URL: {image_url}, status code: {response.status_code}")
            
            # Convert to PIL Image
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Process with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            return image_features.squeeze()
        
        else:
            raise ValueError("Must provide one of: image_path, image_url, or image_tensor")
    
    def get_text_vector(self, description=None, image_path=None, image_url=None):
        """
        Get text embedding for a description, generating one if not provided.
        """
        if description is None:
            if image_url:
                description = self.caption_generator.generate_caption(image_url)
                print(description)
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
        metadata_str, processed_dict = self.metadata_processor.process_metadata(metadata)
        
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
        
        Returns:
            Dictionary with vectors and generated caption (if applicable)
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
        
        # Track if we generate a caption
        generated_caption = None
        
        # Get text vector
        if "description" in item and item["description"]:
            text_vector = self.get_text_vector(description=item["description"])
        elif "image_path" in item:
            generated_caption = self.caption_generator.generate_caption("placeholder")
            text_vector = self.get_text_vector(description=generated_caption)
        elif "image_url" in item:
            generated_caption = self.caption_generator.generate_caption(item["image_url"])
            print(generated_caption)
            text_vector = self.get_text_vector(description=generated_caption)
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

        print("image_vector", image_vector)
        print("text_vector", text_vector)
        print("metadata_vector", metadata_vector)
        print("combined_vector", combined_vector)
        
        result = {
            "image_vector": image_vector.detach().cpu().numpy(),
            "text_vector": text_vector.detach().cpu().numpy(),
            "metadata_vector": metadata_vector.detach().cpu().numpy(),
            "combined_vector": combined_vector.detach().cpu().numpy()
        }
        
        # Include the generated caption in the result if we generated one
        if generated_caption:
            result["generated_caption"] = generated_caption
        
        return result
    
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