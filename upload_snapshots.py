import requests
import os

def upload_snapshots():
    # Qdrant server URL
    base_url = "http://localhost:6333"

    # Define the snapshots to upload
    snapshots = [
        {
            "collection_name": "cosmos_users",
            "snapshot_name": "Cosmos Users Snapshot Apr 1 2025",
            "file_path": "Cosmos Users Snapshot Apr 1 2025.snapshot"
        },
        {
            "collection_name": "cosmos_elements",
            "snapshot_name": "Cosmos Elements Snapshot Apr 1 2025",
            "file_path": "Cosmos Elements Snapshot Apr 1 2025.snapshot"
        }
    ]

    # Upload each snapshot
    for snapshot in snapshots:
        try:
            print(f"Uploading {snapshot['snapshot_name']} to collection {snapshot['collection_name']}...")
            
            # Upload the snapshot file using HTTP API
            upload_url = f"{base_url}/collections/{snapshot['collection_name']}/snapshots/{snapshot['snapshot_name']}"
            with open(snapshot['file_path'], 'rb') as f:
                response = requests.put(upload_url, data=f)
                response.raise_for_status()
            
            print(f"Successfully uploaded {snapshot['snapshot_name']}")
        except Exception as e:
            print(f"Error uploading {snapshot['snapshot_name']}: {str(e)}")

if __name__ == "__main__":
    upload_snapshots() 