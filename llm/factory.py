import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from fastembed import TextEmbedding
from typing import List, Any
from langchain_core.embeddings import Embeddings
from enum import Enum

load_dotenv()

# Enum for LLM providers
class LLMEnums(Enum):
    GOOGLE_GENAI = "GOOGLE_GENAI"
    GROQ = "GROQ"
    
class EmbeddingEnums(Enum):
    LOCAL_EMBEDDING = "LOCAL_EMBEDDING" 

class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self.model.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding_generator = self.model.embed([text])
        embedding = list(embedding_generator)[0]
        return embedding.tolist()
    
    def encode(self, sentences: Any, **kwargs) -> Any:
        if isinstance(sentences, str):
            sentences = [sentences]
        return [e.tolist() for e in self.model.embed(sentences, **kwargs)]

class LLMFactory:
    @staticmethod
    def get_llm():
        """
        Returns an LLM instance based on the LLM_PROVIDER environment variable.
        Supported providers: 'gemini', 'groq'.
        """
        provider = os.getenv("LLM_PROVIDER")
        model_id = os.getenv("LLM_MODEL_ID")

        if not provider:
            raise ValueError("LLM_PROVIDER not found in .env")
        if not model_id:
            raise ValueError("LLM_MODEL_ID not found in .env")

        if provider == LLMEnums.GOOGLE_GENAI.value:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in .env")
            return ChatGoogleGenerativeAI(
                model=model_id, 
                google_api_key=api_key,

            )
        
        elif provider == LLMEnums.GROQ.value:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in .env")
            return ChatGroq(
                model_name=model_id, 
                groq_api_key=api_key
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def get_embedding_model():
        """
        Returns an embedding model instance based on the EMBEDDING_PROVIDER environment variable.
        Supported providers: 'fastembed'.
        """
        provider = os.getenv("EMBEDDING_PROVIDER")
        model_id = os.getenv("EMBEDDING_MODEL_ID")

        if not provider:
            raise ValueError("EMBEDDING_PROVIDER not found in .env")
        if not model_id:
            raise ValueError("EMBEDDING_MODEL_ID not found in .env")

        if provider == EmbeddingEnums.LOCAL_EMBEDDING.value:
            return FastEmbedEmbeddings(model_name=model_id)
        else:
            raise ValueError(f"Unsupported Embedding provider: {provider}")