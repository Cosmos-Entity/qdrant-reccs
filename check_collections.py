from qdrant_client import QdrantClient
import json

def check_collection(client, collection_name):
    """Check details of a collection"""
    print(f"\nChecking collection: {collection_name}")
    
    # Get collection info
    collection_info = client.get_collection(collection_name)
    print(f"Collection config:")
    print(json.dumps(collection_info.dict(), indent=2))
    
    # Count points
    count = client.count(collection_name=collection_name)
    print(f"\nTotal points: {count.count}")
    
    # Get sample points
    points = client.scroll(
        collection_name=collection_name,
        limit=5,
        with_payload=True,
        with_vectors=True
    )[0]
    
    print(f"\nSample points ({len(points)} points):")
    for point in points:
        print(f"\nPoint ID: {point.id}")
        print(f"Vector dimension: {len(point.vector)}")
        print(f"Payload: {point.payload}")

def main():
    client = QdrantClient("localhost", port=6333)
    
    # Check both collections
    check_collection(client, "cosmos_users")
    check_collection(client, "cosmos_elements")

if __name__ == "__main__":
    main() 