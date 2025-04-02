import os
import tarfile
import json
import numpy as np
from qdrant_client import QdrantClient

def examine_segment(segment_path):
    """Examine the contents of a segment file"""
    print(f"\nExamining segment: {os.path.basename(segment_path)}")
    
    with tarfile.open(segment_path, 'r') as tar:
        # List all files in the tar
        print("\nFiles in segment:")
        for member in tar.getmembers():
            print(f"- {member.name}")
            
        # Try to find and read vector data
        vector_files = [m for m in tar.getmembers() if 'vector_storage' in m.name]
        for vector_file in vector_files:
            print(f"\nReading vector file: {vector_file.name}")
            f = tar.extractfile(vector_file)
            if f:
                data = f.read()
                print(f"File size: {len(data)} bytes")
                
                # Try to read as numpy array
                try:
                    # Skip header (first 16 bytes)
                    vector_data = np.frombuffer(data[16:], dtype=np.float32)
                    print(f"Vector data shape: {vector_data.shape}")
                    print(f"First few values: {vector_data[:10]}")
                except Exception as e:
                    print(f"Error reading vector data: {e}")

def main():
    # Path to the segments directory
    segments_dir = "extracted_users_data/0/segments"
    
    # Examine each segment file
    for filename in os.listdir(segments_dir):
        if filename.endswith('.tar'):
            segment_path = os.path.join(segments_dir, filename)
            examine_segment(segment_path)

if __name__ == "__main__":
    main() 