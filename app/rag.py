# app/rag.py
"""
Simple RAG helper using ChromaDB + sentence-transformers.
- Embeds files in data/ (directory.txt, schedules.txt, concerns.txt)
- Stores vectors in ./chroma_db (persistent)
- Usage:
    from rag import Rag
    rag = Rag(build_if_empty=True)    # builds DB from data/ if empty
    hits = rag.search("library hours", n_results=3)
"""

from sentence_transformers import SentenceTransformer
import chromadb
import os
import math
import hashlib
from typing import List

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "school"
EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_FOLDER = "data"
CHUNK_SIZE = 250   # chars per chunk (150-300 recommended)
OVERLAP = 50       # chars overlap between chunks


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n")
    if len(text) <= chunk_size:
        return [text.strip()]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _id_for(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class Rag:
    def __init__(self, build_if_empty: bool = False):
        self.emb = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        # create/get collection
        try:
            self.col = self.client.get_collection(COLLECTION_NAME)
        except Exception:
            self.col = self.client.create_collection(COLLECTION_NAME)
        if build_if_empty and self._is_collection_empty():
            self.build_from_data_folder()

    def _is_collection_empty(self) -> bool:
        try:
            # quick check: get count via small query
            meta = self.col.count()
            return meta == 0
        except Exception:
            return True

    def build_from_data_folder(self, file_list: List[str] = None):
        """
        Read files from data/ and add them to the DB.
        If file_list provided, only those files are used (relative to DATA_FOLDER).
        """
        files = file_list or sorted(
            [f for f in os.listdir(DATA_FOLDER) if os.path.isfile(os.path.join(DATA_FOLDER, f))]
        )
        for fname in files:
            path = os.path.join(DATA_FOLDER, fname)
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read().strip()
            if not text:
                continue
            chunks = _chunk_text(text)
            docs = []
            embeddings = []
            ids = []
            metas = []
            for i, c in enumerate(chunks):
                docs.append(c)
                emb = self.emb.encode(c).tolist()
                embeddings.append(emb)
                _id = _id_for(fname + ":" + str(i))
                ids.append(_id)
                metas.append({"source_file": fname, "chunk_index": i})
            # add to collection
            try:
                self.col.add(
                    documents=docs,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metas
                )
            except Exception as e:
                # fallback: if collection expects different args, try minimal add
                self.col.add(documents=docs, ids=ids, metadatas=metas)
        # ensure data persisted
        try:
            self.client.persist()
        except Exception:
            pass

    def add_text(self, text: str, source: str = "inline"):
        chunks = _chunk_text(text)
        docs, embeddings, ids, metas = [], [], [], []
        for i, c in enumerate(chunks):
            docs.append(c)
            embeddings.append(self.emb.encode(c).tolist())
            ids.append(_id_for(source + ":" + str(i)))
            metas.append({"source": source, "chunk_index": i})
        self.col.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
        try:
            self.client.persist()
        except Exception:
            pass

    def search(self, query: str, n_results: int = 3) -> List[str]:
        """
        Returns a list of matching document chunks (plain text), highest score first.
        """
        q_emb = self.emb.encode(query).tolist()
        results = self.col.query(query_embeddings=[q_emb], n_results=n_results)
        # results structure: {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]]}
        docs = []
        try:
            docs = results["documents"][0]
        except Exception:
            # fallback: older/newer api shapes
            if isinstance(results, list) and len(results) > 0:
                docs = results[0].get("documents", [])
        return docs

    def clear(self):
        """Remove all vectors from the collection."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.col = self.client.create_collection(COLLECTION_NAME)
        try:
            self.client.persist()
        except Exception:
            pass


if __name__ == "__main__":
    # quick usage: build DB from data/ if empty
    r = Rag(build_if_empty=True)
    print("RAG DB ready. Use r.search('your query') to test.")
