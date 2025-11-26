from sentence_transformers import SentenceTransformer
import chromadb
import os
import hashlib
import pypdf
from typing import List

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "school"
EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_FOLDER = "data"

CHUNK_SIZE = 500   
OVERLAP = 100         

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """
    SMART CHUNKER: Splits by 'blocks' (double newlines) first.
    This keeps headers (LOCATION:) attached to their contents.
    """
    # 1. Normalize line endings
    text = text.replace("\r\n", "\n")
    
    # 2. Split by empty lines (Double Newline = New Block)
    blocks = text.split("\n\n")
    
    chunks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        # A) If the block fits in one chunk, keep it whole
        if len(block) < chunk_size:
            chunks.append(block)
        
        # B) If block is huge, split it the old way
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
        print("⏳ Loading RAG Model...")
        self.emb = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        try:
            self.col = self.client.get_collection(COLLECTION_NAME)
        except Exception:
            self.col = self.client.create_collection(COLLECTION_NAME)
            
        if build_if_empty and self._is_collection_empty():
            print("📂 Indexing data folder...")
            self.build_from_data_folder()
            print("✅ RAG Indexing Complete.")

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
            
            # PDF SUPPORT
            if fname.lower().endswith(".pdf"):
                try:
                    print(f"   📄 Processing PDF: {fname}")
                    reader = pypdf.PdfReader(path)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                except Exception as e:
                    print(f"   ❌ Error reading PDF {fname}: {e}")
                    continue
            
            # TEXT SUPPORT
            else:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        text = fh.read().strip()
                except UnicodeDecodeError:
                    print(f"   ⚠️  Skipping binary file: {fname}")
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
                    print(f"   ❌ Error adding {fname} to DB: {e}")

    def search(self, query: str, n_results: int = 3) -> List[str]:
        q_emb = self.emb.encode(query).tolist()
        results = self.col.query(query_embeddings=[q_emb], n_results=n_results)
        docs = []
        try:
            docs = results["documents"][0]
        except Exception:
            if isinstance(results, list) and len(results) > 0:
                docs = results[0].get("documents", [])
        return docs