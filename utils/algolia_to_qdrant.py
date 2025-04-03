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

class AlgoliaToQdrantMigrator:
    """
    Migrate data from Algolia to Qdrant with three-vector embeddings.
    """
    def __init__(
        self,
        algolia_app_id,
        algolia_api_key,
        algolia_index_name,
        qdrant_url,
        qdrant_api_key,
        qdrant_collection,
        batch_size=10,
        openai_api_key=None,
        clip_model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        text_model_name="BAAI/bge-large-en-v1.5",
        llm_model="o1"
    ):
        # Initialize Algolia client
        self.algolia_client = SearchClientSync(algolia_app_id, algolia_api_key)
        self.algolia_index_name = algolia_index_name
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.qdrant_collection = qdrant_collection
        
        # Ensure collection exists
        self._ensure_collection_exists()
        
        # Initialize embedding models
        self.three_vector_embedding = ThreeVectorEmbedding(
            qdrant_url=qdrant_url,
            qdrant_collection=qdrant_collection,
            qdrant_api_key=qdrant_api_key,
            clip_model_name=clip_model_name,
            text_model_name=text_model_name,
            openai_api_key=openai_api_key
        )
        
        # Keep the metadata processor for creating payload
        self.metadata_processor = MetadataProcessor()
        
        # Batch size for processing
        self.batch_size = batch_size
        
        # Stats
        self.processed_count = 0
        self.error_count = 0
        self.start_time = time.time()
    
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
                    "combined": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    "image": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    "text": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    "metadata": models.VectorParams(size=1024, distance=models.Distance.COSINE)
                }
            )
            logger.info(f"Created new collection: {self.qdrant_collection}")
    
    def process_element(self, element):
        """
        Process a single element from Algolia and upload to Qdrant.
        
        Args:
            element: Element data from Algolia
            
        Returns:
            Success status
        """
        try:
            element_id = element.get('objectID')
            if not element_id:
                logger.warning("Element missing objectID, skipping")
                return False
            
            # Extract image URL from the nested image object
            image_url = None
            if 'image' in element and isinstance(element['image'], dict) and 'url' in element['image']:
                image_url = element['image']['url']
            elif 'imageUrl' in element:
                image_url = element['imageUrl']
            
            if not image_url:
                logger.warning(f"Element {element_id} missing image URL, skipping")
                return False
            
            if not image_url.endswith("?format=jpeg"):
                image_url = f"{image_url}?format=jpeg"
            
            # Get description if available
            description = element.get('description', None)
            
            logger.info(f"Processing element {element_id} with image URL: {image_url}")
            
            # Process metadata for payload - get both string and dict
            metadata_str, processed_metadata = self.metadata_processor.process_metadata(element)
            
            try:
                # Generate caption first so we can capture it
                generated_caption = None
                if description is None:
                    # Only generate caption if no description is provided
                    try:
                        generated_caption = self.three_vector_embedding.caption_generator.generate_caption(image_url)
                        logger.info(f"Generated caption for {element_id}: {generated_caption[:100]}...")
                    except Exception as e:
                        logger.warning(f"Error generating caption for {element_id}: {e}")
                
                # Create item dictionary for ThreeVectorEmbedding.process_item
                item = {
                    "image_url": image_url,
                    "description": generated_caption,  # Use the generated caption if available
                    "metadata": element  # Pass the entire element as metadata
                }
                
                # Process the item using ThreeVectorEmbedding
                vectors = self.three_vector_embedding.process_item(item)
                
                # Prepare payload - combine original data with processed metadata
                payload = {
                    "metadata_str": metadata_str,
                    "processed_at": time.time()
                }
                
                # Add the generated caption to the payload if available
                if generated_caption:
                    payload["generated_caption"] = generated_caption
                
                # Add all processed metadata fields directly to the payload
                # This makes them directly queryable in Qdrant
                for key, value in processed_metadata.items():
                    payload[key] = value
                
                # Upload to Qdrant
                self.qdrant_client.upsert(
                    collection_name=self.qdrant_collection,
                    points=[
                        models.PointStruct(
                            id=int(element_id),
                            vector={
                                "combined": vectors["combined_vector"],
                                "image": vectors["image_vector"],
                                "text": vectors["text_vector"],
                                "metadata": vectors["metadata_vector"]
                            },
                            payload=payload
                        )
                    ]
                )
                
                logger.info(f"Successfully processed and uploaded element {element_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error processing element {element_id}: {e}")
                self.error_count += 1
                return False
            
        except Exception as e:
            logger.error(f"Error processing element {element.get('objectID', 'unknown')}: {e}")
            self.error_count += 1
            return False
    
    def migrate(self, limit=None, start_id=1):
        """
        Migrate data from Algolia to Qdrant.
        
        Args:
            limit: Maximum number of elements to process (None for all)
            start_id: Starting ID to process
            
        Returns:
            Number of elements processed
        """
        logger.info(f"Starting migration from Algolia to Qdrant")
        
        if limit:
            logger.info(f"Processing up to {limit} elements starting from ID {start_id}")
            
            # Process elements by ID
            requests = []
            for i in range(start_id, start_id + limit):
                requests.append({
                    "objectID": str(i),
                    "indexName": self.algolia_index_name
                })
            
            # Get elements from Algolia
            response = self.algolia_client.get_objects(
                get_objects_params={
                    "requests": requests
                }
            )
            
            # Extract results and filter out None values (IDs that don't exist)
            results = response.to_dict()["results"]
            elements = [result for result in results if result is not None]
            
            if not elements:
                logger.warning("No elements found in Algolia")
                return 0
            
            logger.info(f"Found {len(elements)} elements to process")
            
            # Process each element
            with tqdm(total=len(elements), desc="Migrating") as pbar:
                for element in elements:
                    if self.process_element(element):
                        self.processed_count += 1
                    
                    pbar.update(1)
        
        else:
            # Get total count using search
            total_count_response = self.algolia_client.search(
                {
                    "requests": [
                        {
                            "indexName": self.algolia_index_name,
                            "query": "",
                            "params": {
                                "hitsPerPage": 0
                            }
                        }
                    ]
                }
            )
            total_count = total_count_response.to_dict()["results"][0]["nbHits"]
            
            logger.info(f"Found {total_count} total elements in Algolia")
            
            # Process in batches
            page = 0
            processed = 0
            
            with tqdm(total=total_count, desc="Migrating") as pbar:
                while processed < total_count:
                    # Get batch from Algolia
                    batch_size = min(self.batch_size, total_count - processed)
                    results = self.algolia_client.search(
                        {
                            "requests": [
                                {
                                    "indexName": self.algolia_index_name,
                                    "query": "",
                                    "params": {
                                        "hitsPerPage": batch_size,
                                        "page": page
                                    }
                                }
                            ]
                        }
                    )
                    
                    hits = results.to_dict()["results"][0]["hits"]
                    
                    if not hits:
                        logger.info("No more results from Algolia")
                        break
                    
                    # Process each element
                    for element in hits:
                        if self.process_element(element):
                            self.processed_count += 1
                        
                        processed += 1
                        pbar.update(1)
                    
                    page += 1
                    
                    # Print stats
                    elapsed = time.time() - self.start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {processed}/{total_count} elements ({rate:.2f}/s), errors: {self.error_count}")
        
        # Final stats
        elapsed = time.time() - self.start_time
        logger.info(f"Migration complete: {self.processed_count} elements processed in {elapsed:.2f}s")
        logger.info(f"Errors: {self.error_count}")
        
        return self.processed_count

def main():
    parser = argparse.ArgumentParser(description="Migrate data from Algolia to Qdrant with three-vector embeddings")
    
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of elements to process")
    parser.add_argument("--start", type=int, default=1,
                        help="Starting ID to process (when using --limit)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size for processing")
    parser.add_argument("--qdrant-collection", type=str, default="cosmos_elements_dev_v2",
                        help="Name of Qdrant collection")
    parser.add_argument("--llm-model", type=str, default="o1",
                        help="OpenAI model to use for captioning")
    parser.add_argument("--by-id", action="store_true",
                        help="Process elements by ID instead of using search")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials from environment
    algolia_app_id = os.environ.get("ALGOLIA_APP_ID")
    algolia_api_key = os.environ.get("ALGOLIA_API_KEY")
    algolia_index_name = "SingleElements"  # Updated to use SingleElements
    
    qdrant_url = os.environ.get("QDRANT_API_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    qdrant_collection = os.environ.get("QDRANT_MAIN_COLLECTION_NAME")
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Initialize migrator
    migrator = AlgoliaToQdrantMigrator(
        algolia_app_id=algolia_app_id,
        algolia_api_key=algolia_api_key,
        algolia_index_name=algolia_index_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection=qdrant_collection,
        batch_size=args.batch_size,
        openai_api_key=openai_api_key,
        llm_model=args.llm_model
    )
    
    # Run migration
    if args.by_id or args.limit is not None:
        migrator.migrate(limit=args.limit, start_id=args.start)
    else:
        migrator.migrate()

if __name__ == "__main__":
    main() 