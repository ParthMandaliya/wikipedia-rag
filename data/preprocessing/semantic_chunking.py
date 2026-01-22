from typing import Optional
from langchain_experimental.text_splitter import SemanticChunker
from embeddings import Embedder

class SemanticChunkerSingleton:
    """Singleton that returns SemanticChunker instance"""
    
    _chunker: Optional[SemanticChunker] = None
    
    def __new__(
        cls,
        embedder: Embedder,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: int = 85
    ) -> SemanticChunker: 
        
        if cls._chunker is None:
            cls._chunker = SemanticChunker(
                embeddings=embedder._model,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount,
            )
        
        return cls._chunker 
