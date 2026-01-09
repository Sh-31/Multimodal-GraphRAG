import os
import uuid
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from llm.factory import LLMFactory

class QdrantVectorDB:
    def __init__(self, collection_name: str = "multimodal_rag"):
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY")
        
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key
        )
        
        self.collection_name = collection_name
        self.embedding_model = LLMFactory.get_embedding_model()
        self._ensure_collection()

    def _get_embedding_size(self) -> int:
        return len(self.embedding_model.embed_query("test"))

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self._get_embedding_size(),
                    distance=models.Distance.COSINE
                )
            )
            
    def index_chunks(self, chunks: List[str], metadatas: List[Dict[str, Any]] = None, ids: List[str] = None):
        """
        Index chunks of text.
        """
        if not chunks:
            return
            
        embeddings = self.embedding_model.embed_documents(chunks)
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]
            
        points = []
        for i, (chunk, embedding, id_) in enumerate(zip(chunks, embeddings, ids)):
            metadata = metadatas[i] if metadatas else {}
            metadata["text"] = chunk
            
            points.append(
                models.PointStruct(
                    id=id_,
                    vector=embedding,
                    payload=metadata
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return f"Indexed {len(chunks)} chunks into Qdrant collection '{self.collection_name}'"

    def search(self, query: str, limit: int = 5):
        embedding = self.embedding_model.embed_query(query)
     
        # Use new query_points API for qdrant-client >= 1.12.0
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
        )
        
        return response.points

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)