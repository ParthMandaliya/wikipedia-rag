import warnings
from pathlib import Path
from typing import Dict, Any, Optional

from datasets import IterableDataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from data import get_hotpotqa
from embeddings import Embedder
from data.preprocessing import (
    SemanticChunkerSingleton, RecursiveChunkerSingleton
)
from vector_db import WikipediaVectorStore, create_vector_db
from schemas import Config


def main():
    config: Config = Config.from_yaml("./config.yaml")
    
    # embedding configs
    device: str = config.embedding.device
    backend: str = config.embedding.backend

    # chunking configs
    skip_n_articles: int = config.chunking.skip_articles
    if skip_n_articles > 0:
        warnings.warn(
            f"First {skip_n_articles:,} articles will be skipped...",
            category=UserWarning
        )

    process_n_articles: Optional[int] = config.chunking.process_articles
    if process_n_articles is not None:
        warnings.warn(
            f"Only first {process_n_articles} will be processed...",
            category=UserWarning
        )

    chunk_type: str = config.chunking.type
    breakpoint_threshold_amount: int = config.chunking.percentile
    
    # vector db configs
    vector_db_collection_name: str = config.vectordb.collection_name
    vector_db_save_path: Path = Path(config.vectordb.db_path)

    hotpotqa_ds: IterableDataset = get_hotpotqa(
        name="fullwiki", split="train",
    )
    embedding_model = Embedder(device=device, backend=backend)

    if chunk_type == "semantic":
        chunker: SemanticChunker = SemanticChunkerSingleton(
            embedding_model,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )
    else:
        # only used for RecursiveCharacterTextSplitter
        chunk_size: int = config["chunking"]["chunk_size"]
        overlap: int = config["chunking"]["overlap"]
        
        chunker: RecursiveCharacterTextSplitter = RecursiveChunkerSingleton(
            chunk_size=chunk_size, chunk_overlap=overlap,
        )
    
    vector_db = WikipediaVectorStore(
        embedder=embedding_model,
        db_save_path=vector_db_save_path,
        collection_name=vector_db_collection_name,
    )

    if config.vectordb.create_vector_db:
        create_vector_db(
            hotpotqad_ds=hotpotqa_ds,
            chunking_type=chunk_type,
            chunker=chunker,
            vector_db=vector_db,
            skip_n_articles=skip_n_articles,
            process_n_articles=process_n_articles,
        )
 

if __name__ == "__main__":
    main()
