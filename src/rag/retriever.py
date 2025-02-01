from typing import List, Dict, Optional
from .embeddings import OllamaEmbeddings
from .vector_store import FAISSVectorStore
import re


class ResumeRetriever:
    def __init__(self):
        """Initialize the resume retriever with embedding model and vector store."""
        self.embeddings = OllamaEmbeddings()
        self.dimension = self.embeddings.get_embedding_dimension()
        if not self.dimension:
            raise ValueError("Failed to initialize embeddings dimension")
        self.vector_store = FAISSVectorStore(self.dimension)

    def _chunk_resume(self, text: str) -> List[Dict]:
        """Split resume into meaningful chunks with metadata."""
        chunks = []
        current_section = None
        current_text = []

        # Simple section detection
        section_pattern = r'^\\section{(.+?)}'
        subsection_pattern = r'\\resumeSubheading{(.+?)}{(.+?)}{(.+?)}{(.+?)}'

        for line in text.split('\n'):
            # Check for new section
            section_match = re.match(section_pattern, line)
            if section_match:
                # Save previous section if exists
                if current_section and current_text:
                    chunks.append({
                        'text': '\n'.join(current_text),
                        'metadata': {'section': current_section}
                    })
                current_section = section_match.group(1)
                current_text = [line]
                continue

            # Check for subsections (like work experience entries)
            subsection_match = re.match(subsection_pattern, line)
            if subsection_match and current_section:
                # Save previous subsection if exists
                if current_text:
                    chunks.append({
                        'text': '\n'.join(current_text),
                        'metadata': {'section': current_section}
                    })
                current_text = [line]
                continue

            current_text.append(line)

        # Add final chunk
        if current_section and current_text:
            chunks.append({
                'text': '\n'.join(current_text),
                'metadata': {'section': current_section}
            })

        return chunks

    def index_resume(self, resume_text: str):
        """Process and index resume content."""
        # Chunk the resume
        chunks = self._chunk_resume(resume_text)

        # Get embeddings for all chunks
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)

        # Add to vector store
        self.vector_store.add_texts(
            texts=texts,
            embeddings=embeddings,
            metadatas=[chunk['metadata'] for chunk in chunks]
        )

    def get_relevant_experience(
        self,
        job_description: str,
        num_chunks: int = 3
    ) -> List[Dict]:
        """Retrieve relevant experience based on job description."""
        # Get embedding for job description
        query_embedding = self.embeddings.embed_query(job_description)
        if not query_embedding:
            raise ValueError(
                "Failed to generate embedding for job description")

        # Search for relevant chunks
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=num_chunks
        )

        return results

    def save(self, directory: str):
        """Save the indexed resume data."""
        self.vector_store.save(directory)

    @classmethod
    def load(cls, directory: str) -> 'ResumeRetriever':
        """Load a previously saved resume retriever."""
        retriever = cls()
        retriever.vector_store = FAISSVectorStore.load(directory)
        return retriever
