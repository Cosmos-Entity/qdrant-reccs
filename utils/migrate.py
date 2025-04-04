import os
import argparse
import time
from dotenv import load_dotenv
from algoliasearch.search.client import SearchClientSync
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import sys
import concurrent.futures
import torch
import asyncio
import aiohttp
import json
from functools import partial
from typing import List, Dict, Any, Tuple, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
import threading
import openai

# Get absolute path to the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the beginning of sys.path to ensure it's checked first
sys.path.insert(0, parent_dir)

# Import directly from the parent module using absolute import
import three_vector_embedding
from three_vector_embedding import (
    LLMCaptionGenerator, 
    TextEmbeddingModel, 
    MetadataProcessor, 
    ThreeVectorEmbedding,
    ThreeVectorCombiner
)
from utils.image_utils import url_to_base64, download_image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define weight combinations for different vector strategies
WEIGHT_COMBINATIONS = [
    (0.33, 0.33, 0.34),  # Equal weights
    (0.6, 0.2, 0.2),     # Image-focused
    (0.2, 0.6, 0.2),     # Text-focused
    (0.2, 0.2, 0.6)      # Metadata-focused
]

def combine_vectors_with_weights(image_vector, text_vector, metadata_vector, weights):
    """
    Combine vectors using specified weights.
    """
    # Convert to numpy arrays if they're not already
    if isinstance(image_vector, torch.Tensor):
        image_vector = image_vector.detach().cpu().numpy()
    if isinstance(text_vector, torch.Tensor):
        text_vector = text_vector.detach().cpu().numpy()
    if isinstance(metadata_vector, torch.Tensor):
        metadata_vector = metadata_vector.detach().cpu().numpy()
    
    # Combine vectors using weights
    image_weight, text_weight, metadata_weight = weights
    combined = (
        image_weight * image_vector +
        text_weight * text_vector +
        metadata_weight * metadata_vector
    )
    
    # Normalize the combined vector
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    
    return combined

class OptimizedThreeVectorEmbedding(ThreeVectorEmbedding):
    """
    Optimized version of ThreeVectorEmbedding with preloaded models, batching, and multi-threading
    """
    def __init__(self, **kwargs):
        # Extract our custom parameters before passing the rest to the parent class
        self.clip_batch_size = kwargs.pop('clip_batch_size', 16)
        self.max_workers = kwargs.pop('max_workers', 8)
        
        # Filter out any other parameters that might not be accepted by the parent class
        parent_params = {}
        allowed_params = [
            'clip_model_name', 'text_model_name', 'openai_api_key', 
            'fixed_weights', 'vector_dim', 'qdrant_url', 
            'qdrant_collection', 'qdrant_api_key'
        ]
        
        for param in allowed_params:
            if param in kwargs:
                parent_params[param] = kwargs[param]
        
        # Initialize parent class with only the allowed parameters
        super().__init__(**parent_params)
        
        # Preload models once to avoid reloading for each item
        logger.info("Preloading models...")
        
        # Ensure text embedder is loaded
        if hasattr(self.text_embedder, '_ensure_model_loaded'):
            self.text_embedder._ensure_model_loaded()
            
            # Move text model to GPU if available
            if torch.cuda.is_available() and hasattr(self.text_embedder, 'model'):
                logger.info("Moving text model to GPU")
                self.text_embedder.model = self.text_embedder.model.to('cuda')
        
        # Preload CLIP model to GPU if available
        if torch.cuda.is_available():
            logger.info("Moving CLIP model to GPU")
            self.clip_model = self.clip_model.to('cuda')
            # Set mixed precision for faster inference
            self.mixed_precision = True
            
            # Enable torch.compile for PyTorch 2.0+ if available
            if hasattr(torch, 'compile') and callable(getattr(torch, 'compile')):
                try:
                    logger.info("Optimizing CLIP model with torch.compile()")
                    self.clip_model = torch.compile(self.clip_model)
                except Exception as e:
                    logger.warning(f"Could not compile model: {e}")
        else:
            self.mixed_precision = False
        
        # Create image download cache to avoid re-downloading images
        self.image_cache = {}
        self.cache_lock = threading.Lock()
        
        logger.info("Models preloaded successfully")
    
    def get_image_vector_batch(self, image_urls: List[str]):
        """
        Process a batch of images in parallel to get CLIP embeddings
        
        Optimized with:
        1. Parallel image downloading
        2. Multi-threaded processing
        3. Efficient GPU batching
        4. Mixed precision for faster computation
        5. Image caching to avoid re-downloading
        """
        if not image_urls:
            return []
        
        start_time = time.time()
        logger.info(f"Processing {len(image_urls)} images in optimized batch mode")
        
        # Step 1: Preprocess URLs and check cache
        normalized_urls = [
            url if url.endswith("?format=jpeg") else f"{url}?format=jpeg" 
            for url in image_urls
        ]
        
        # Step 2: Find which images need downloading
        cached_vectors = {}
        urls_to_download = []
        
        with self.cache_lock:
            for url in normalized_urls:
                if url in self.image_cache:
                    cached_vectors[url] = self.image_cache[url]
                else:
                    urls_to_download.append(url)
        
        logger.info(f"Cache hits: {len(cached_vectors)}/{len(image_urls)}")
        
        # Step 3: Download and process images in parallel
        downloaded_images = {}
        if urls_to_download:
            downloaded_images = self._download_images_parallel(urls_to_download)
        
        # Step 4: Process images with CLIP in optimized batches
        all_vectors = {}
        
        # Add cached vectors
        all_vectors.update(cached_vectors)
        
        # Process downloaded images in efficient GPU batches
        if downloaded_images:
            batch_vectors = self._process_images_in_batches(downloaded_images)
            all_vectors.update(batch_vectors)
            
            # Update cache with new vectors
            with self.cache_lock:
                for url, vector in batch_vectors.items():
                    if url not in self.image_cache:
                        self.image_cache[url] = vector
        
        # Step 5: Arrange vectors in original URL order
        vectors = []
        for url in image_urls:
            norm_url = url if url.endswith("?format=jpeg") else f"{url}?format=jpeg"
            if norm_url in all_vectors:
                vectors.append(all_vectors[norm_url])
            else:
                logger.warning(f"Missing vector for {url}, using zero vector")
                vectors.append(np.zeros(1024))
        
        elapsed = time.time() - start_time
        logger.info(f"Processed {len(image_urls)} images in {elapsed:.2f}s ({len(image_urls)/elapsed:.2f} img/s)")
        
        return vectors
    
    def _download_images_parallel(self, urls: List[str]) -> Dict[str, torch.Tensor]:
        """Download multiple images in parallel using a thread pool"""
        images = {}
        
        def download_single_image(url):
            try:
                from PIL import Image
                from io import BytesIO
                
                # Download the image
                response = requests.get(url, stream=True, timeout=10)
                if response.status_code != 200:
                    logger.error(f"Failed to download {url}, status: {response.status_code}")
                    return url, None
                
                # Convert to PIL Image
                image = Image.open(BytesIO(response.content)).convert('RGB')
                
                # Process with CLIP processor but don't move to GPU yet (will do in batches)
                inputs = self.clip_processor(images=image, return_tensors="pt")
                
                return url, inputs.pixel_values[0]  # Return the preprocessed tensor
                
            except Exception as e:
                logger.error(f"Error downloading image {url}: {e}")
                return url, None
        
        # Use thread pool for parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(download_single_image, urls))
        
        # Collect successful downloads
        for url, tensor in results:
            if tensor is not None:
                images[url] = tensor
        
        logger.info(f"Downloaded {len(images)}/{len(urls)} images successfully")
        return images
    
    def _process_images_in_batches(self, image_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Process images in optimal GPU batches using mixed precision"""
        if not image_dict:
            return {}
        
        urls = list(image_dict.keys())
        tensors = list(image_dict.values())
        vectors = {}
        
        # Process in batches to maximize GPU utilization
        for i in range(0, len(tensors), self.clip_batch_size):
            batch_urls = urls[i:i+self.clip_batch_size]
            batch_tensors = tensors[i:i+self.clip_batch_size]
            
            # Stack tensors into a batch
            batch_tensor = torch.stack(batch_tensors)
            
            # Move batch to GPU if available
            if torch.cuda.is_available():
                batch_tensor = batch_tensor.to('cuda')
            
            # Use mixed precision for faster computation
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                with torch.no_grad():
                    batch_features = self.clip_model.get_image_features(pixel_values=batch_tensor)
            
            # Move back to CPU and convert to numpy
            batch_features = batch_features.detach().cpu().numpy()
            
            # Store results
            for j, url in enumerate(batch_urls):
                vectors[url] = batch_features[j]
        
        return vectors
    
    def get_text_vector_batch(self, descriptions: List[str]):
        """
        Process a batch of descriptions to get text embeddings
        Optimized for GPU processing and larger batches
        """
        if not descriptions:
            return []
        
        start_time = time.time()
        
        try:
            # Filter out empty descriptions
            valid_indices = []
            valid_descriptions = []
            
            for i, desc in enumerate(descriptions):
                if desc and isinstance(desc, str):
                    valid_indices.append(i)
                    valid_descriptions.append(desc)
            
            if not valid_descriptions:
                logger.warning("No valid descriptions found, returning zero vectors")
                return [np.zeros(1024) for _ in descriptions]
            
            # Process in batches if there are many descriptions
            batch_size = 64  # Larger batch size for text processing
            all_embeddings = []
            
            for i in range(0, len(valid_descriptions), batch_size):
                batch_descriptions = valid_descriptions[i:i+batch_size]
                
                # Get batch embeddings with GPU acceleration
                with torch.cuda.amp.autocast(enabled=self.mixed_precision and torch.cuda.is_available()):
                    batch_embeddings = self.text_embedder.get_embeddings_batch(batch_descriptions)
                
                # Move to CPU and convert to numpy
                batch_embeddings = batch_embeddings.detach().cpu().numpy()
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all batches
            if len(all_embeddings) > 1:
                valid_embeddings = np.vstack(all_embeddings)
            else:
                valid_embeddings = all_embeddings[0]
            
            # Create result array with zero vectors for invalid descriptions
            result = [np.zeros(1024) for _ in descriptions]
            for i, idx in enumerate(valid_indices):
                result[idx] = valid_embeddings[i]
            
            elapsed = time.time() - start_time
            logger.info(f"Processed {len(descriptions)} text embeddings in {elapsed:.2f}s ({len(descriptions)/elapsed:.2f} texts/s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in batch text embedding: {e}")
            # Return zero vectors as fallback
            return [np.zeros(1024) for _ in descriptions]
    
    def get_metadata_vector_batch(self, metadata_list: List[Dict]):
        """
        Process a batch of metadata dictionaries
        Optimized for parallel processing
        """
        if not metadata_list:
            return []
        
        start_time = time.time()
        
        # Process each metadata to string format in parallel
        metadata_strings = []
        processed_dicts = []
        
        def process_single_metadata(metadata):
            try:
                metadata_str, processed_dict = self.metadata_processor.process_metadata(metadata)
                return metadata_str, processed_dict
            except Exception as e:
                logger.error(f"Error processing metadata: {e}")
                return "", {}
        
        # Use thread pool for parallel metadata processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_single_metadata, metadata_list))
        
        # Collect results
        for metadata_str, processed_dict in results:
            metadata_strings.append(metadata_str)
            processed_dicts.append(processed_dict)
        
        # Get embeddings for all strings
        vectors = self.get_text_vector_batch(metadata_strings)
        
        elapsed = time.time() - start_time
        logger.info(f"Processed {len(metadata_list)} metadata embeddings in {elapsed:.2f}s ({len(metadata_list)/elapsed:.2f} items/s)")
        
        return vectors
    
    def process_batch(self, batch_items: List[Dict]) -> List[Dict]:
        """
        Process a batch of items in parallel to get all vectors
        
        Returns a list of dictionaries with vectors and captions
        """
        if not batch_items:
            return []
        
        start_time = time.time()
        logger.info(f"Processing batch of {len(batch_items)} items")
        
        # Extract inputs for batch processing
        image_urls = []
        descriptions = []
        metadata_list = []
        
        for item in batch_items:
            if "image_url" in item:
                image_urls.append(item["image_url"])
            elif "image_path" in item:
                # For local files, you might need a different approach
                image_urls.append(None)
            else:
                image_urls.append(None)
            
            descriptions.append(item.get("description"))
            metadata_list.append(item.get("metadata", {}))
        
        # Process all vectors in parallel
        image_vectors = self.get_image_vector_batch(image_urls)
        text_vectors = self.get_text_vector_batch(descriptions)
        metadata_vectors = self.get_metadata_vector_batch(metadata_list)
        
        # Combine vectors
        results = []
        for i, item in enumerate(batch_items):
            image_vector = image_vectors[i]
            text_vector = text_vectors[i]
            metadata_vector = metadata_vectors[i]
            
            # Combine vectors using weights
            combined_vector = self.get_combined_vector(
                torch.tensor(image_vector), 
                torch.tensor(text_vector), 
                torch.tensor(metadata_vector)
            ).detach().cpu().numpy()
            
            results.append({
                "image_vector": image_vector,
                "text_vector": text_vector,
                "metadata_vector": metadata_vector,
                "combined_vector": combined_vector,
                "generated_caption": item.get("generated_caption")
            })
        
        elapsed = time.time() - start_time
        logger.info(f"Processed batch in {elapsed:.2f}s ({len(batch_items)/elapsed:.2f} items/s)")
        
        return results

class LightningFastCaptionGenerator:
    """
    Lightning-fast async caption generator with advanced batching, caching, and rate limiting strategies
    """
    def __init__(self, api_key=None, model="o1", batch_size=10, max_concurrent=20, use_server_batching=True):
        import openai
        from cachetools import LRUCache
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as environment variable")
        
        self.model = model
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.use_server_batching = use_server_batching
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Cache to avoid re-processing the same images
        # Store up to 1000 image URLs and their captions
        self.caption_cache = LRUCache(maxsize=1000)
        
        # Semaphore to control maximum concurrent API calls
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Track rate limits for adaptive throttling
        self.rate_limit_tracker = {
            'remaining': 100,  # Assumed initial value
            'reset_at': time.time() + 60,  # Assumed reset time
            'total': 100  # Assumed total rate limit
        }
        
        # Same prompt as the original, but slightly condensed for speed
        self.prompt_template = """
        You are a direct and insightful image reviewer. Determine if the image is:
        1. Professional Art/Photo: Provide 2-3 sentences on subject, style, composition, color palette, and technical quality.
        2. Casual image: Briefly describe what's visible in 1-2 sentences.
        Keep descriptions factual and concise.
        """
        
        # For batch processing, create a function call definition
        self.batch_function_def = {
            "name": "process_image_batch",
            "description": "Process a batch of images and provide captions for each",
            "parameters": {
                "type": "object",
                "properties": {
                    "captions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Caption for each image in the batch"
                        }
                    }
                },
                "required": ["captions"]
            }
        }
        
        logger.info(f"Initialized LightningFastCaptionGenerator with batch_size={batch_size}, max_concurrent={max_concurrent}")
    
    async def generate_captions_async(self, image_urls: List[str]) -> List[str]:
        """
        Generate captions for multiple images using optimal batching strategy
        """
        if not image_urls:
            return []
        
        start_time = time.time()
        logger.info(f"Generating captions for {len(image_urls)} images")
        
        # Check cache first and only process uncached URLs
        cached_results = {}
        uncached_urls = []
        
        for url in image_urls:
            norm_url = self._normalize_url(url)
            if norm_url in self.caption_cache:
                cached_results[url] = self.caption_cache[norm_url]
            else:
                uncached_urls.append(url)
        
        logger.info(f"Cache hit: {len(cached_results)}/{len(image_urls)} images already cached")
        
        # If all images were cached, return immediately
        if not uncached_urls:
            logger.info(f"All images were cached, returning cached results")
            return [cached_results[url] for url in image_urls]
        
        # Process uncached URLs
        results = {}
        
        # Determine optimal processing strategy based on batch size
        if self.use_server_batching and len(uncached_urls) <= self.batch_size:
            # Use server-side batching for small batches
            logger.info(f"Using server-side batching for {len(uncached_urls)} images")
            batch_results = await self._process_server_batch(uncached_urls)
            for url, caption in zip(uncached_urls, batch_results):
                norm_url = self._normalize_url(url)
                self.caption_cache[norm_url] = caption
                results[url] = caption
        else:
            # Process in parallel with controlled concurrency for larger batches
            logger.info(f"Using parallel processing with semaphore for {len(uncached_urls)} images")
            tasks = []
            
            # Process in client-side batches to avoid overwhelming resources
            for i in range(0, len(uncached_urls), self.batch_size):
                sub_batch = uncached_urls[i:i+self.batch_size]
                if len(sub_batch) > 1 and self.use_server_batching:
                    # Use server-side batching for sub-batches
                    tasks.append(self._process_batch_with_semaphore(sub_batch))
                else:
                    # Process individually for smaller batches
                    for url in sub_batch:
                        tasks.append(self._process_url_with_semaphore(url))
            
            # Wait for all tasks to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results and handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    continue
                    
                if isinstance(result, dict):
                    # Single URL result
                    url = result['url']
                    caption = result['caption']
                    norm_url = self._normalize_url(url)
                    self.caption_cache[norm_url] = caption
                    results[url] = caption
                elif isinstance(result, list) and all(isinstance(r, dict) for r in result):
                    # Batch result
                    for r in result:
                        url = r['url']
                        caption = r['caption']
                        norm_url = self._normalize_url(url)
                        self.caption_cache[norm_url] = caption
                        results[url] = caption
        
        # Combine cached and new results in original order
        final_results = []
        for url in image_urls:
            if url in cached_results:
                final_results.append(cached_results[url])
            elif url in results:
                final_results.append(results[url])
            else:
                # Fallback for any URLs that failed
                final_results.append(None)
        
        elapsed = time.time() - start_time
        rate = len(image_urls) / elapsed if elapsed > 0 else 0
        logger.info(f"Generated {len(results)} captions in {elapsed:.2f}s ({rate:.2f} images/s)")
        
        return final_results
    
    async def _process_batch_with_semaphore(self, urls: List[str]) -> List[Dict]:
        """Use semaphore to control concurrent batch requests"""
        async with self.semaphore:
            return await self._process_server_batch(urls)
    
    async def _process_url_with_semaphore(self, url: str) -> Dict:
        """Use semaphore to control concurrent individual requests"""
        async with self.semaphore:
            try:
                caption = await self._generate_single_caption(url)
                return {'url': url, 'caption': caption}
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                return {'url': url, 'caption': None}
    
    async def _process_server_batch(self, urls: List[str]) -> List[Dict]:
        """
        Process a batch of URLs using server-side batching
        This uses function calling to process multiple images in a single API call
        """
        try:
            # Prepare all the images for batching
            norm_urls = [self._normalize_url(url) for url in urls]
            base64_urls = [url_to_base64(url) for url in norm_urls]
            
            # Create content array with all images
            content = [{"type": "text", "text": "Please caption each of the following images:"}]
            for i, base64_url in enumerate(base64_urls):
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": base64_url}
                })
                # Add a separator between images
                if i < len(base64_urls) - 1:
                    content.append({"type": "text", "text": f"Image {i+1} above, Image {i+2} below:"})
            
            # Make single API call with all images
            response = await self._make_api_call_with_retry(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_template},
                    {"role": "user", "content": content}
                ],
                functions=[self.batch_function_def],
                function_call={"name": "process_image_batch"}
            )
            
            # Extract results from function call
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "process_image_batch":
                try:
                    args = json.loads(function_call.arguments)
                    captions = args.get("captions", [])
                    
                    # Ensure we have the right number of captions
                    if len(captions) != len(urls):
                        logger.warning(f"Received {len(captions)} captions but expected {len(urls)}")
                        # Pad with None if needed
                        captions = captions + [None] * (len(urls) - len(captions))
                    
                    return [{'url': url, 'caption': caption} for url, caption in zip(urls, captions)]
                    
                except Exception as e:
                    logger.error(f"Error parsing function call results: {e}")
                    # Fall back to processing individually
                    return await self._process_urls_individually(urls)
            else:
                # If function calling fails, extract from the text response
                text_response = response.choices[0].message.content
                captions = self._parse_batch_response(text_response, len(urls))
                return [{'url': url, 'caption': caption} for url, caption in zip(urls, captions)]
                
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Fall back to processing individually
            return await self._process_urls_individually(urls)
    
    async def _process_urls_individually(self, urls: List[str]) -> List[Dict]:
        """Process URLs individually as fallback"""
        results = []
        for url in urls:
            try:
                caption = await self._generate_single_caption(url)
                results.append({'url': url, 'caption': caption})
            except Exception as e:
                logger.error(f"Error processing individual URL {url}: {e}")
                results.append({'url': url, 'caption': None})
        return results
    
    def _parse_batch_response(self, text_response: str, expected_count: int) -> List[str]:
        """Parse a text response that contains multiple captions"""
        # Simple parsing logic - look for numbered sections or distinct paragraphs
        captions = []
        
        # Try to find numbered sections like "1.", "2.", etc.
        import re
        numbered_sections = re.split(r'\n\s*\d+[\.\)]\s*', '\n' + text_response)
        
        if len(numbered_sections) > 1:
            # Skip the first empty section before the first number
            captions = numbered_sections[1:expected_count+1]
        else:
            # Fall back to paragraph splitting
            paragraphs = [p.strip() for p in text_response.split('\n\n') if p.strip()]
            captions = paragraphs[:expected_count]
        
        # Ensure we have the expected number of captions
        if len(captions) < expected_count:
            captions = captions + [None] * (expected_count - len(captions))
        
        return captions[:expected_count]  # Trim if we have too many
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _generate_single_caption(self, image_url: str) -> str:
        """
        Generate a caption for a single image
        """
        norm_url = self._normalize_url(image_url)
        base64_url = url_to_base64(norm_url)
        
        # Make API call
        response = await self._make_api_call_with_retry(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt_template},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": base64_url}}
                ]}
            ]
        )
        
        return response.choices[0].message.content
    
    async def _make_api_call_with_retry(self, **kwargs):
        """Make API call with adaptive rate limiting and retry logic"""
        # Check if we need to wait for rate limit reset
        now = time.time()
        if self.rate_limit_tracker['remaining'] < 5 and now < self.rate_limit_tracker['reset_at']:
            wait_time = self.rate_limit_tracker['reset_at'] - now + 1  # Add 1 second buffer
            logger.warning(f"Rate limit nearly exceeded. Waiting {wait_time:.2f}s for reset")
            await asyncio.sleep(wait_time)
        
        try:
            # Make the API call
            response = self.client.chat.completions.create(**kwargs)
            
            # Update rate limit tracking from headers if available
            # OpenAI doesn't consistently provide these in the client, but if they did:
            # self.rate_limit_tracker['remaining'] = int(response.headers.get('x-ratelimit-remaining', 100))
            # self.rate_limit_tracker['reset_at'] = time.time() + int(response.headers.get('x-ratelimit-reset', 60))
            
            return response
            
        except openai.RateLimitError as e:
            # Adjust our tracking on rate limit errors
            self.rate_limit_tracker['remaining'] = 0
            self.rate_limit_tracker['reset_at'] = time.time() + 60  # Assume 1 minute reset
            logger.warning(f"Rate limit exceeded: {e}")
            # Let the retry decorator handle the retry
            raise
            
        except Exception as e:
            logger.error(f"API call error: {e}")
            raise
    
    def _normalize_url(self, url: str) -> str:
        """Ensure URL ends with ?format=jpeg"""
        if not url.endswith("?format=jpeg"):
            return f"{url}?format=jpeg"
        return url

# For backward compatibility
FastCaptionGenerator = LightningFastCaptionGenerator

class OptimizedAlgoliaToQdrantMigrator:
    """
    Optimized migration from Algolia to Qdrant with parallel processing
    """
    def __init__(
        self,
        algolia_app_id,
        algolia_api_key,
        algolia_index_name,
        qdrant_url,
        qdrant_api_key,
        qdrant_collection,
        batch_size=64,  # Increased batch size
        openai_api_key=None,
        clip_model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        text_model_name="BAAI/bge-large-en-v1.5",
        llm_model="o1",
        num_workers=16,  # Default number of parallel workers
        skip_captions=False,  # Option to skip caption generation
        caption_batch_size=10,  # Batch size for caption generation
        clip_batch_size=16,  # Batch size for CLIP processing
        use_gpu=True,  # Use GPU for acceleration
        max_concurrent_llm=20,  # Maximum concurrent LLM requests
        checkpoint_file="migration_checkpoint.json",  # File to store progress
        resume=False  # Whether to resume from checkpoint
    ):
        # Initialize Algolia client
        self.algolia_client = SearchClientSync(algolia_app_id, algolia_api_key)
        self.algolia_index_name = algolia_index_name
        
        # Initialize Qdrant client with optimized settings
        self.qdrant_client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key,
            timeout=60,  # Increased timeout for larger batches
            prefer_grpc=True  # Use gRPC for better performance when available
        )
        self.qdrant_collection = qdrant_collection
        
        # Ensure collection exists
        self._ensure_collection_exists()
        
        # Processing parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.skip_captions = skip_captions
        self.caption_batch_size = caption_batch_size
        self.clip_batch_size = clip_batch_size
        self.checkpoint_file = checkpoint_file
        self.resume = resume
        
        # Load checkpoint if resuming
        self.processed_ids = set()
        if self.resume and os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    self.processed_ids = set(checkpoint_data.get('processed_ids', []))
                    logger.info(f"Resuming from checkpoint with {len(self.processed_ids)} processed items")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
        
        # Extract only the parameters that ThreeVectorEmbedding expects
        three_vector_params = {
            'qdrant_url': qdrant_url,
            'qdrant_collection': qdrant_collection,
            'qdrant_api_key': qdrant_api_key,
            'clip_model_name': clip_model_name,
            'text_model_name': text_model_name,
            'openai_api_key': openai_api_key,
        }
        
        # Add our custom parameters separately
        three_vector_params['clip_batch_size'] = clip_batch_size
        three_vector_params['max_workers'] = num_workers
        
        # Initialize embedding models with optimized parameters
        self.three_vector_embedding = OptimizedThreeVectorEmbedding(**three_vector_params)
        
        # Initialize lightning-fast caption generator
        if not self.skip_captions:
            self.caption_generator = LightningFastCaptionGenerator(
                api_key=openai_api_key,
                model=llm_model,
                batch_size=caption_batch_size,
                max_concurrent=max_concurrent_llm,
                use_server_batching=True
            )
        
        # Keep the metadata processor for creating payload
        self.metadata_processor = MetadataProcessor()
        
        # Stats
        self.processed_count = len(self.processed_ids)
        self.error_count = 0
        self.start_time = time.time()
        
        # Performance tracking
        self.batch_times = []
        self.caption_times = []
        self.vector_times = []
        self.upload_times = []
        
        # GPU setting
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.info("Running in CPU mode")
    
    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists."""
        try:
            # Check if collection exists
            self.qdrant_client.get_collection(self.qdrant_collection)
            logger.info(f"Using existing collection: {self.qdrant_collection}")
        except Exception:
            # Create collection if it doesn't exist
            self.qdrant_client.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config={
                    "combined_equal": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    "combined_image": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    "combined_text": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    "combined_metadata": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    "image": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    "text": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    "metadata": models.VectorParams(size=1024, distance=models.Distance.COSINE)
                }
            )
            logger.info(f"Created new collection: {self.qdrant_collection}")
    
    async def process_batch(self, elements):
        """
        Process a batch of elements in parallel with optimal performance
        """
        if not elements:
            return 0
        
        batch_start_time = time.time()
        
        # Extract element IDs and image URLs
        element_ids = []
        image_urls = []
        descriptions = []
        all_metadata = []
        
        # Skip already processed elements if resuming
        for element in elements:
            element_id = element.get('objectID')
            if not element_id:
                logger.warning("Element missing objectID, skipping")
                continue
            
            # Skip if already processed
            if str(element_id) in self.processed_ids:
                logger.debug(f"Skipping already processed element {element_id}")
                continue
            
            # Extract image URL
            image_url = None
            if 'image' in element and isinstance(element['image'], dict) and 'url' in element['image']:
                image_url = element['image']['url']
            elif 'imageUrl' in element:
                image_url = element['imageUrl']
            
            if not image_url:
                logger.warning(f"Element {element_id} missing image URL, skipping")
                continue
            
            # Get description if available
            description = element.get('description', None)
            
            element_ids.append(element_id)
            image_urls.append(image_url)
            descriptions.append(description)
            all_metadata.append(element)
        
        if not element_ids:
            return 0
        
        logger.info(f"Processing batch of {len(element_ids)} elements")
        
        # Step 1: Generate captions in parallel if needed
        caption_start_time = time.time()
        generated_captions = [None] * len(element_ids)
        
        if not self.skip_captions:
            caption_needed_indices = []
            caption_needed_urls = []
            
            for i, description in enumerate(descriptions):
                if description is None:
                    caption_needed_indices.append(i)
                    caption_needed_urls.append(image_urls[i])
            
            if caption_needed_urls:
                logger.info(f"Generating captions for {len(caption_needed_urls)} images")
                batch_captions = await self.caption_generator.generate_captions_async(caption_needed_urls)
                
                # Assign captions to the correct indices
                for i, caption in zip(caption_needed_indices, batch_captions):
                    generated_captions[i] = caption
        
        caption_time = time.time() - caption_start_time
        self.caption_times.append(caption_time)
        
        # Step 2: Process all vectors in parallel
        vector_start_time = time.time()
        
        # Prepare items for batch processing
        batch_items = []
        for i, element_id in enumerate(element_ids):
            item = {
                "image_url": image_urls[i],
                "description": descriptions[i] if descriptions[i] else generated_captions[i],
                "metadata": all_metadata[i],
                "generated_caption": generated_captions[i]
            }
            batch_items.append(item)
        
        # Process vectors in optimized batch mode
        processed_results = self.three_vector_embedding.process_batch(batch_items)
        
        vector_time = time.time() - vector_start_time
        self.vector_times.append(vector_time)
        logger.info(f"Vector processing complete in {vector_time:.2f}s")
        
        # Step 3: Prepare points for Qdrant
        upload_start_time = time.time()
        points = []
        
        for i, element_id in enumerate(element_ids):
            try:
                result = processed_results[i]
                
                image_vector = result["image_vector"]
                text_vector = result["text_vector"]
                metadata_vector = result["metadata_vector"]
                
                # Create combined vectors with different weight strategies
                combined_vectors = {
                    "combined_equal": combine_vectors_with_weights(
                        image_vector, text_vector, metadata_vector, WEIGHT_COMBINATIONS[0]
                    ),
                    "combined_image": combine_vectors_with_weights(
                        image_vector, text_vector, metadata_vector, WEIGHT_COMBINATIONS[1]
                    ),
                    "combined_text": combine_vectors_with_weights(
                        image_vector, text_vector, metadata_vector, WEIGHT_COMBINATIONS[2]
                    ),
                    "combined_metadata": combine_vectors_with_weights(
                        image_vector, text_vector, metadata_vector, WEIGHT_COMBINATIONS[3]
                    ),
                    "image": image_vector,
                    "text": text_vector,
                    "metadata": metadata_vector
                }
                
                # Process metadata for payload
                metadata_str, processed_metadata = self.metadata_processor.process_metadata(all_metadata[i])
                
                # Prepare payload
                payload = {
                    "metadata_str": metadata_str,
                    "processed_at": time.time(),
                    "weight_combinations": {
                        "equal": WEIGHT_COMBINATIONS[0],
                        "image_focused": WEIGHT_COMBINATIONS[1],
                        "text_focused": WEIGHT_COMBINATIONS[2],
                        "metadata_focused": WEIGHT_COMBINATIONS[3]
                    }
                }
                
                # Add the generated caption to the payload if available
                if result.get("generated_caption"):
                    payload["generated_caption"] = result["generated_caption"]
                
                # Add all processed metadata fields directly to the payload
                for key, value in processed_metadata.items():
                    payload[key] = value
                
                # Create point for Qdrant
                points.append(
                    models.PointStruct(
                        id=int(element_id),
                        vector=combined_vectors,
                        payload=payload
                    )
                )
                
            except Exception as e:
                logger.error(f"Error processing element {element_id}: {e}")
                self.error_count += 1
        
        # Step 4: Upload points to Qdrant in batches
        if points:
            # Process in larger batches for better throughput
            qdrant_batch_size = 200  # Increased batch size for better performance
            
            # Use concurrent uploads for multiple batches
            upload_tasks = []
            
            for j in range(0, len(points), qdrant_batch_size):
                sub_batch = points[j:j+qdrant_batch_size]
                
                # Define upload task
                async def upload_batch(batch):
                    try:
                        # Use upsert to handle existing points
                        self.qdrant_client.upsert(
                            collection_name=self.qdrant_collection,
                            points=batch,
                            wait=True
                        )
                        return len(batch)
                    except Exception as e:
                        logger.error(f"Error uploading batch to Qdrant: {e}")
                        self.error_count += len(batch)
                        return 0
                
                upload_tasks.append(upload_batch(sub_batch))
            
            # Wait for all upload tasks to complete
            upload_results = await asyncio.gather(*upload_tasks)
            uploaded_count = sum(upload_results)
            
            # Update processed IDs for checkpoint
            for element_id in element_ids:
                self.processed_ids.add(str(element_id))
            
            # Save checkpoint
            self._save_checkpoint()
            
            upload_time = time.time() - upload_start_time
            self.upload_times.append(upload_time)
            logger.info(f"Uploaded {uploaded_count} points to Qdrant in {upload_time:.2f}s")
        
        batch_time = time.time() - batch_start_time
        self.batch_times.append(batch_time)
        
        # Print performance metrics
        if len(self.batch_times) >= 5:
            avg_batch_time = sum(self.batch_times[-5:]) / 5
            avg_caption_time = sum(self.caption_times[-5:]) / 5 if self.caption_times[-5:] else 0
            avg_vector_time = sum(self.vector_times[-5:]) / 5
            avg_upload_time = sum(self.upload_times[-5:]) / 5
            
            logger.info(f"Performance metrics (last 5 batches):")
            logger.info(f"  Avg batch time: {avg_batch_time:.2f}s")
            if not self.skip_captions:
                logger.info(f"  Avg caption time: {avg_caption_time:.2f}s ({avg_caption_time/avg_batch_time*100:.1f}%)")
            logger.info(f"  Avg vector time: {avg_vector_time:.2f}s ({avg_vector_time/avg_batch_time*100:.1f}%)")
            logger.info(f"  Avg upload time: {avg_upload_time:.2f}s ({avg_upload_time/avg_batch_time*100:.1f}%)")
        
        return len(points)
    
    def _save_checkpoint(self):
        """Save progress checkpoint to disk"""
        try:
            checkpoint_data = {
                'processed_ids': list(self.processed_ids),
                'processed_count': len(self.processed_ids),
                'error_count': self.error_count,
                'last_updated': time.time()
            }
            
            # Use atomic write pattern to prevent corruption
            temp_file = f"{self.checkpoint_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(checkpoint_data, f)
            
            # Rename to final file
            os.replace(temp_file, self.checkpoint_file)
            
            logger.debug(f"Saved checkpoint with {len(self.processed_ids)} processed IDs")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    async def migrate(self, limit=None, start_id=1):
        """
        Migrate data from Algolia to Qdrant with async processing
        """
        logger.info(f"Starting lightning-fast migration from Algolia to Qdrant")
        logger.info(f"GPU: {'Enabled' if self.use_gpu else 'Disabled'}, Workers: {self.num_workers}, Batch size: {self.batch_size}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        try:
            # First, try to get a sample batch to determine if we can access the index
            sample_requests = [{"objectID": str(i), "indexName": self.algolia_index_name} for i in range(1, 11)]
            sample_response = self.algolia_client.get_objects(get_objects_params={"requests": sample_requests})
            sample_results = sample_response.to_dict()["results"]
            valid_sample_results = [result for result in sample_results if result is not None]
            
            logger.info(f"Successfully retrieved {len(valid_sample_results)} sample items from Algolia")
            
            if limit:
                logger.info(f"Processing up to {limit} elements starting from ID {start_id}")
                
                # Process elements by ID in batches
                total_processed = 0
                while total_processed < limit:
                    current_batch_size = min(self.batch_size, limit - total_processed)
                    current_start = start_id + total_processed
                    
                    # Prepare requests for this batch
                    requests = []
                    for i in range(current_start, current_start + current_batch_size):
                        # Skip if already processed
                        if str(i) in self.processed_ids:
                            continue
                            
                        requests.append({
                            "objectID": str(i),
                            "indexName": self.algolia_index_name
                        })
                    
                    if not requests:
                        logger.info("All requested IDs already processed, finishing")
                        break
                    
                    # Get elements from Algolia
                    try:
                        response = self.algolia_client.get_objects(
                            get_objects_params={"requests": requests}
                        )
                        
                        # Extract results and filter out None values (IDs that don't exist)
                        results = response.to_dict()["results"]
                        elements = [result for result in results if result is not None]
                        
                        if not elements:
                            logger.warning(f"No elements found for IDs {current_start}-{current_start + current_batch_size - 1}")
                            total_processed += current_batch_size
                            continue
                        
                        logger.info(f"Found {len(elements)}/{len(requests)} elements to process")
                        
                        # Process batch
                        processed = await self.process_batch(elements)
                        self.processed_count += processed
                        total_processed += current_batch_size
                        
                        # Print progress
                        elapsed = time.time() - self.start_time
                        rate = self.processed_count / elapsed if elapsed > 0 else 0
                        logger.info(f"Progress: {total_processed}/{limit} elements, {self.processed_count} points created ({rate:.2f}/s)")
                        
                    except Exception as e:
                        logger.error(f"Error processing batch starting at ID {current_start}: {e}")
                        total_processed += current_batch_size
            else:
                # Without a limit, we need to iterate through all objects
                # We'll use a paging approach with get_objects in batches
                page_size = self.batch_size
                current_page = 0
                total_processed = 0
                continue_processing = True
                
                while continue_processing:
                    # Create a batch of sequential IDs to try
                    start_id = current_page * page_size + 1
                    end_id = start_id + page_size - 1
                    
                    requests = []
                    for i in range(start_id, end_id + 1):
                        if str(i) not in self.processed_ids:
                            requests.append({
                                "objectID": str(i),
                                "indexName": self.algolia_index_name
                            })
                    
                    if not requests:
                        logger.info(f"No new IDs to process in range {start_id}-{end_id}, trying next batch")
                        current_page += 1
                        
                        # Safety check to avoid infinite loop
                        if current_page > 1000:  # Arbitrary limit
                            logger.warning("Reached maximum page limit, stopping")
                            break
                        continue
                    
                    try:
                        response = self.algolia_client.get_objects(
                            get_objects_params={"requests": requests}
                        )
                        
                        results = response.to_dict()["results"]
                        elements = [result for result in results if result is not None]
                        
                        if not elements:
                            logger.info(f"No valid elements found in range {start_id}-{end_id}, trying next batch")
                            current_page += 1
                            
                            # If we've had several empty batches in a row, we might be at the end
                            if current_page % 5 == 0:
                                logger.info(f"Processed {total_processed} elements so far with {self.processed_count} points created")
                            continue
                        
                        logger.info(f"Found {len(elements)}/{len(requests)} elements to process in range {start_id}-{end_id}")
                        
                        # Process batch
                        processed = await self.process_batch(elements)
                        self.processed_count += processed
                        total_processed += len(elements)
                        
                        # Print progress
                        elapsed = time.time() - self.start_time
                        rate = self.processed_count / elapsed if elapsed > 0 else 0
                        logger.info(f"Processed {total_processed} elements so far, {self.processed_count} points created ({rate:.2f}/s)")
                        
                        current_page += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing batch at page {current_page}: {e}")
                        current_page += 1  # Try to continue with next page
                        await asyncio.sleep(5)  # Wait a bit before retrying
        
        except Exception as e:
            logger.error(f"Error in Algolia migration: {e}")
            raise
        
        # Final stats
        elapsed = time.time() - self.start_time
        logger.info(f"Migration complete: {self.processed_count} elements processed in {elapsed:.2f}s")
        logger.info(f"Overall rate: {self.processed_count/elapsed:.2f} elements/s")
        logger.info(f"Errors: {self.error_count}")
        
        # Print performance breakdown
        if self.batch_times:
            avg_batch_time = sum(self.batch_times) / len(self.batch_times)
            avg_caption_time = sum(self.caption_times) / len(self.caption_times) if self.caption_times else 0
            avg_vector_time = sum(self.vector_times) / len(self.vector_times) if self.vector_times else 0
            avg_upload_time = sum(self.upload_times) / len(self.upload_times) if self.upload_times else 0
            
            logger.info(f"Performance breakdown:")
            logger.info(f"  Avg batch time: {avg_batch_time:.2f}s")
            if not self.skip_captions:
                logger.info(f"  Avg caption time: {avg_caption_time:.2f}s ({avg_caption_time/avg_batch_time*100:.1f}% of total)")
            logger.info(f"  Avg vector time: {avg_vector_time:.2f}s ({avg_vector_time/avg_batch_time*100:.1f}% of total)")
            logger.info(f"  Avg upload time: {avg_upload_time:.2f}s ({avg_upload_time/avg_batch_time*100:.1f}% of total)")
            
            if not self.skip_captions:
                logger.info(f"Caption generation rate: {len(self.caption_times) / sum(self.caption_times):.2f} captions/s")
        
        return self.processed_count

async def main_async():
    parser = argparse.ArgumentParser(description="Lightning-fast migration from Algolia to Qdrant with three-vector embeddings")
    
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of elements to process")
    parser.add_argument("--start", type=int, default=1,
                        help="Starting ID to process (when using --limit)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for processing")
    parser.add_argument("--qdrant-collection", type=str, default="cosmos_elements_dev_v2",
                        help="Name of Qdrant collection")
    parser.add_argument("--llm-model", type=str, default="o1",
                        help="OpenAI model to use for captioning")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers")
    parser.add_argument("--skip-captions", action="store_true",
                        help="Skip caption generation with LLM")
    parser.add_argument("--caption-batch-size", type=int, default=10,
                        help="Batch size for caption generation")
    parser.add_argument("--clip-batch-size", type=int, default=16,
                        help="Batch size for CLIP processing")
    parser.add_argument("--max-concurrent-llm", type=int, default=20,
                        help="Maximum concurrent LLM requests")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    parser.add_argument("--checkpoint-file", type=str, default="migration_checkpoint.json",
                        help="File to store progress")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--profile", action="store_true",
                        help="Enable performance profiling")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials from environment
    algolia_app_id = os.environ.get("ALGOLIA_APP_ID")
    algolia_api_key = os.environ.get("ALGOLIA_API_KEY")
    algolia_index_name = "SingleElements"
    
    qdrant_url = os.environ.get("QDRANT_API_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    qdrant_collection = args.qdrant_collection or os.environ.get("QDRANT_MAIN_COLLECTION_NAME")
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Enable profiling if requested
    if args.profile:
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()
    
    try:
        # Initialize lightning-fast migrator
        migrator = OptimizedAlgoliaToQdrantMigrator(
            algolia_app_id=algolia_app_id,
            algolia_api_key=algolia_api_key,
            algolia_index_name=algolia_index_name,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_collection=qdrant_collection,
            batch_size=args.batch_size,
            openai_api_key=openai_api_key,
            llm_model=args.llm_model,
            num_workers=args.workers,
            skip_captions=args.skip_captions,
            caption_batch_size=args.caption_batch_size,
            clip_batch_size=args.clip_batch_size,
            max_concurrent_llm=args.max_concurrent_llm,
            use_gpu=not args.no_gpu,
            checkpoint_file=args.checkpoint_file,
            resume=args.resume
        )
        
        # Run migration
        await migrator.migrate(limit=args.limit, start_id=args.start)
        
    finally:
        # Print profiling results if enabled
        if args.profile:
            pr.disable()
            stats = pstats.Stats(pr)
            stats.sort_stats('cumtime')
            stats.print_stats(30)  # Print top 30 functions by cumulative time

def main():
    # Run the async main function
    asyncio.run(main_async())

if __name__ == "__main__":
    main()