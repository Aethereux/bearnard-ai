import chromadb

# Point to your persistent chroma database
client = chromadb.PersistentClient(path="chroma_db")

# List all collections
collections = client.list_collections()
print("\n✅ Collections in ChromaDB:")
for coll in collections:
    print("-", coll.name)

# Assuming your collection is called "school_data"
print("\n📂 Reading items from 'school' collection...")
try:
    collection = client.get_collection(name="school")

    data = collection.get(include=["documents", "metadatas", "embeddings"])

    print(f"\nTotal items: {len(data['ids'])}\n")

    for i in range(len(data["ids"])):
        print(f"--- Entry {i+1} ---")
        print("ID:        ", data["ids"][i])
        print("Document:  ", data["documents"][i])
        print("Metadata:  ", data["metadatas"][i])
        print()
except Exception as e:
    print("⚠ Error:", e)
    print("Maybe the collection name is different?")
