from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
print("Available methods:", dir(client)) 