from sentence_transformers import SentenceTransformer
import chromadb
import os
import hashlib
import pypdf
from typing import List

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "school"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2" 
DATA_FOLDER = "data"

CHUNK_SIZE = 1000   
OVERLAP = 200         

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n")
    blocks = text.split("\n\n")
    
    chunks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        if len(block) < chunk_size:
            chunks.append(block)
        else:
            start = 0
            while start < len(block):
                end = start + chunk_size
                sub_chunk = block[start:end].strip()
                if sub_chunk:
                    chunks.append(sub_chunk)
                start = end - overlap
                
    return chunks

def _id_for(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

class Rag:
    def __init__(self, build_if_empty: bool = False):
        print("Loading RAG Model...")
        self.emb = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        try:
            self.col = self.client.get_collection(COLLECTION_NAME)
        except Exception:
            self.col = self.client.create_collection(COLLECTION_NAME)
            
        if build_if_empty and self._is_collection_empty():
            print("Indexing data folder...")
            self.build_from_data_folder()
            print("RAG Indexing Complete.")

    def _is_collection_empty(self) -> bool:
        try:
            meta = self.col.count()
            return meta == 0
        except Exception:
            return True

    def build_from_data_folder(self, file_list: List[str] = None):
        files = file_list or sorted(
            [f for f in os.listdir(DATA_FOLDER) if os.path.isfile(os.path.join(DATA_FOLDER, f))]
        )
        
        for fname in files:
            path = os.path.join(DATA_FOLDER, fname)
            text = ""
            
            if fname.lower().endswith(".pdf"):
                try:
                    print(f"Processing PDF: {fname}")
                    reader = pypdf.PdfReader(path)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                except Exception as e:
                    print(f"Error reading PDF {fname}: {e}")
                    continue
            else:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        text = fh.read().strip()
                except UnicodeDecodeError:
                    print(f"Skipping binary file: {fname}")
                    continue

            if not text:
                continue

            chunks = _chunk_text(text)
            docs = []
            embeddings = []
            ids = []
            metas = []
            
            if chunks:
                chunk_embeddings = self.emb.encode(chunks).tolist()
                for i, c in enumerate(chunks):
                    docs.append(c)
                    embeddings.append(chunk_embeddings[i])
                    _id = _id_for(fname + ":" + str(i))
                    ids.append(_id)
                    metas.append({"source_file": fname, "chunk_index": i})
                
                try:
                    if docs:
                        self.col.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
                except Exception as e:
                    print(f"Error adding {fname} to DB: {e}")

    def search(self, query: str, n_results: int = 15, distance_threshold: float = 1.6) -> List[str]:
        print(f"[DEBUG] Searching for: '{query}'")
        
        q_emb = self.emb.encode(query).tolist()
        results = self.col.query(query_embeddings=[q_emb], n_results=n_results)
        
        docs = []
        distances = []
        
        try:
            docs = results["documents"][0]
            distances = results["distances"][0]
            
            filtered_docs = []
            for i, (doc, dist) in enumerate(zip(docs, distances)):
                if dist <= distance_threshold:
                    filtered_docs.append(doc)

            
            print(f"[RAG] Retrieved {len(filtered_docs)} chunks from DB.")
            return filtered_docs
            
        except Exception as e:
            print(f"[DEBUG] Error in search: {e}")
            return []