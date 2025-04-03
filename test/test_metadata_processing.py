import os
import sys
import argparse
from dotenv import load_dotenv
from algoliasearch.search.client import SearchClientSync
import json
from pprint import pprint

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from three_vector_embedding import MetadataProcessor

def main():
    parser = argparse.ArgumentParser(description="Test metadata processing on Algolia elements")
    parser.add_argument("--limit", type=int, default=5, help="Number of elements to process")
    parser.add_argument("--start", type=int, default=1, help="Starting ID to process")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get Algolia credentials from environment
    algolia_app_id = os.environ.get("ALGOLIA_APP_ID")
    algolia_api_key = os.environ.get("ALGOLIA_API_KEY")
    algolia_index_name = "SingleElements"
    
    if not algolia_app_id or not algolia_api_key:
        print("Error: Algolia credentials not found in environment variables")
        sys.exit(1)
    
    # Initialize Algolia client
    client = SearchClientSync(algolia_app_id, algolia_api_key)
    
    # Prepare requests for get_objects
    requests = []
    for i in range(args.start, args.start + args.limit):
        requests.append({
            "objectID": str(i),  # Convert to string as objectIDs are typically strings
            "indexName": algolia_index_name
        })
    
    # Get elements from Algolia
    response = client.get_objects(
        get_objects_params={
            "requests": requests
        }
    )
    
    # Extract results and filter out None values (IDs that don't exist)
    results = response.to_dict()["results"]
    hits = [result for result in results if result is not None]
    
    if not hits:
        print("No elements found in Algolia")
        sys.exit(1)
    
    print(f"Processing {len(hits)} elements from Algolia\n")
    
    # Initialize metadata processor
    metadata_processor = MetadataProcessor()
    
    # Process each element
    for i, element in enumerate(hits):
        element_id = element.get('objectID', 'unknown')
        print(f"Element {i+1}: {element_id}")
        
        # Print original metadata (all fields)
        print("\nOriginal metadata (all fields):")
        print("-" * 50)
        for field, value in element.items():
            # For large lists, truncate the output
            if isinstance(value, list) and len(value) > 10:
                print(f"  {field}: {value[:10]} ... (and {len(value)-10} more items)")
            # For large strings, truncate the output
            elif isinstance(value, str) and len(value) > 200:
                print(f"  {field}: {value[:200]}... (truncated)")
            # For dictionaries, use pprint for better readability
            elif isinstance(value, dict):
                print(f"  {field}:")
                pprint(value, indent=4, width=100, depth=2)
            else:
                print(f"  {field}: {value}")
        print("-" * 50)
        
        # Process metadata
        metadata_str = metadata_processor.process_metadata(element)
        
        # Print processed metadata string
        print("\nProcessed metadata string:")
        print("=" * 80)
        print(metadata_str)
        print("=" * 80)
        
        print("\n" + "#"*80 + "\n")
    
    print(f"Done! Successfully processed {len(hits)} out of {args.limit} requested elements.")

if __name__ == "__main__":
    main() 