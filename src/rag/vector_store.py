import faiss
import numpy as np
from typing import List, Dict, Optional
import pickle
import os


class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        """Initialize FAISS vector store with specified embedding dimension."""
        self.dimension = embedding_dim
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts: List[str] = []
        self.metadata: List[Dict] = []

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None
    ):
        """Add texts and their embeddings to the vector store."""
        if not texts or not embeddings:
            return

        if metadatas is None:
            metadatas = [{} for _ in texts]

        embeddings_array = np.array(embeddings).astype('float32')

        # Add to FAISS index
        self.index.add(embeddings_array)

        # Store texts and metadata
        self.texts.extend(texts)
        self.metadata.extend(metadatas)

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4
    ) -> List[Dict]:
        """Search for similar texts using query embedding."""
        query_embedding_array = np.array([query_embedding]).astype('float32')

        # Search the index
        distances, indices = self.index.search(query_embedding_array, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                results.append({
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(distances[0][i])
                })

        return results

    def save(self, directory: str):
        """Save the vector store to disk."""
        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        # Save texts and metadata
        with open(os.path.join(directory, "store_data.pkl"), "wb") as f:
            pickle.dump({
                'texts': self.texts,
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)

    @classmethod
    def load(cls, directory: str) -> 'FAISSVectorStore':
        """Load a vector store from disk."""
        # Load texts and metadata
        with open(os.path.join(directory, "store_data.pkl"), "rb") as f:
            data = pickle.load(f)

        # Create instance
        vector_store = cls(data['dimension'])
        vector_store.texts = data['texts']
        vector_store.metadata = data['metadata']

        # Load FAISS index
        vector_store.index = faiss.read_index(
            os.path.join(directory, "index.faiss"))

        return vector_store
