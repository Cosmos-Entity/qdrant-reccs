from qdrant_client import QdrantClient, models
import os
import json
import requests
import shutil

def restore_collection(client, collection_name, snapshot_dir):
    """Restore a collection from a snapshot directory"""
    print(f"Restoring collection {collection_name} from {snapshot_dir}")
    
    # First check if collection exists
    collections = client.get_collections().collections
    exists = any(col.name == collection_name for col in collections)
    
    if exists:
        print(f"Collection {collection_name} already exists, recreating...")
        client.delete_collection(collection_name)
    
    # Read collection config
    with open(os.path.join(snapshot_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Create collection with the same config
    vectors_config = config["params"]["vectors"]
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config
    )
    
    # Copy snapshot to Qdrant's snapshot directory
    qdrant_dir = os.path.expanduser("~/.qdrant")
    snapshots_dir = os.path.join(qdrant_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    
    target_dir = os.path.join(snapshots_dir, collection_name)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(snapshot_dir, target_dir)
    
    # Format path for Qdrant
    formatted_path = f"file://{target_dir.replace(os.sep, '/')}"
    
    # Use HTTP API to restore the snapshot
    restore_url = f"http://localhost:6333/collections/{collection_name}/snapshots/recover"
    response = requests.put(restore_url, json={"location": formatted_path})
    
    if response.status_code == 200:
        print(f"Successfully restored collection {collection_name}")
        return True
    else:
        print(f"Error restoring collection {collection_name}: {response.text}")
        return False

def main():
    # Initialize Qdrant client
    client = QdrantClient("localhost", port=6333)
    
    # Restore both collections
    success = restore_collection(client, "cosmos_users", "extracted_users_data")
    if success:
        success = restore_collection(client, "cosmos_elements", "extracted_elements_data")
    
    if success:
        print("\nAll collections restored successfully!")
        print("You can now use these collections in old_code.py")
    else:
        print("\nError occurred during restoration. Please check the error messages above.")

if __name__ == "__main__":
    main() 