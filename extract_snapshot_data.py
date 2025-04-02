import os
import json
import tarfile
import tempfile
from qdrant_client import QdrantClient

def extract_snapshot_data(snapshot_path, output_dir=None):
    """
    Extract data from a Qdrant snapshot file.
    
    Args:
        snapshot_path: Path to the snapshot file
        output_dir: Directory to extract data to (optional)
    
    Returns:
        Path to the extracted data
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    print(f"Extracting snapshot: {snapshot_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the snapshot (Qdrant snapshots are POSIX tar archives)
    with tarfile.open(snapshot_path, 'r') as tar:
        tar.extractall(path=output_dir)
    
    print(f"Extraction complete. Data extracted to: {output_dir}")
    
    # List the extracted files
    extracted_files = os.listdir(output_dir)
    print(f"Extracted files: {extracted_files}")
    
    return output_dir

def main():
    # Define snapshot paths
    users_snapshot = "Cosmos Users Snapshot Apr 1 2025.snapshot"
    elements_snapshot = "Cosmos Elements Snapshot Apr 1 2025.snapshot"
    
    # Create output directories
    users_output = "extracted_users_data"
    elements_output = "extracted_elements_data"
    
    # Extract data from snapshots
    users_data_dir = extract_snapshot_data(users_snapshot, users_output)
    elements_data_dir = extract_snapshot_data(elements_snapshot, elements_output)
    
    print("\nExtraction complete for both snapshots.")
    print(f"Users data: {users_data_dir}")
    print(f"Elements data: {elements_data_dir}")

if __name__ == "__main__":
    main() 