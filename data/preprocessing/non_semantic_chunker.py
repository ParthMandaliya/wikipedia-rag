from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveChunkerSingleton:
    """Singleton that returns RecursiveCharacterTextSplitter instance"""
    
    _chunker: Optional[RecursiveCharacterTextSplitter] = None
    
    def __new__(
        cls,
        chunk_size: int = 1000,
        chunk_overlap: int = 85
    ) -> RecursiveCharacterTextSplitter: 
        
        if cls._chunker is None:
            cls._chunker = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        
        return cls._chunker
