import subprocess
import json
import numpy as np


class OllamaEmbeddings:
    def __init__(self, model_name="nomic-embed-text:latest"):
        self.model_name = model_name

    def _call_ollama(self, text):
        """Call Ollama API to get embeddings."""
        cmd = [
            "curl",
            "http://localhost:11434/api/embeddings",
            "-d",
            json.dumps({"model": self.model_name, "prompt": text})
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            response = json.loads(result.stdout)
            return response.get('embedding')
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None

    def embed_documents(self, texts):
        """Generate embeddings for a list of documents."""
        embeddings = []
        for text in texts:
            embedding = self._call_ollama(text)
            if embedding:
                embeddings.append(embedding)
        return embeddings

    def embed_query(self, text):
        """Generate embedding for a single query text."""
        return self._call_ollama(text)

    def get_embedding_dimension(self):
        """Get the dimension of the embeddings."""
        # Test with a simple text to get embedding dimension
        test_embedding = self._call_ollama("test")
        if test_embedding:
            return len(test_embedding)
        return None
