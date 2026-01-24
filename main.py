import warnings
from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Set

from datasets import IterableDataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from data import get_wiki, get_unique_titles_from_squad
from embeddings import Embedder
from data.preprocessing import (
    SemanticChunkerSingleton, RecursiveChunkerSingleton, get_chunks
)
from vector_db import WikipediaVectorStore
from utilities import load_config


def create_vector_db(
    wiki_ds: IterableDataset,
    accepted_titles: Set[str],
    chunker: Union[SemanticChunker, RecursiveCharacterTextSplitter],
    vector_db: WikipediaVectorStore,
    # batch_size: Optional[int] = 128,
    skip_n_batches: Optional[int] = 0,
) -> None:
    with tqdm(desc="Downloading first batch...", total=None) as pbar:
        article: Optional[Dict[str, str]] = None
        for i, article in enumerate(wiki_ds):
            if i < skip_n_batches:
                pbar.set_description(f"Skipping batch: {i:,}...")
                continue
            
            pbar.set_description(
                f"Articles: {i:,} | Chunks: {vector_db.total_chunks:,} | Downloading..."
            )

            doc_to_split = Document(
                page_content=article["text"],
                metadata={
                    "article_id": article.get("id", ""),
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                }
            )

            pbar.set_description(
                f"Articles: {i:,} | Chunks: {vector_db.total_chunks:,} | Chunking document..."
            )
            splitted_docs: List[Document] = get_chunks([doc_to_split], chunker)
            
            pbar.set_description(
                f"Articles: {i:,} | Chunks: {vector_db.total_chunks:,} | "
                f"Adding {len(splitted_docs):,} chunks to ChromaDB..."
            )
            vector_db.add_documents(splitted_docs)

            pbar.set_description(
                f"Articles: {i:,} | Chunks: {vector_db.total_chunks:,} | "
                f"Added {len(splitted_docs):,} chunks to ChromaDB..."
            )
            del splitted_docs

            pbar.update(1)

            if i >= len(accepted_titles):
                break
        
        pbar.set_description(
            f"Complete! Articles Processed: {i+1:,} | "
            f"Total Chunks Stored in ChromaDB: {vector_db.total_chunks:,}"
        )

    print(f"Final vector store saved at: {vector_db.db_save_path}")
    print(f"Total articles processed: {i+1:,}")
    print(f"Total chunks indexed: {vector_db.total_chunks:,}")


def main():
    config: Dict[str, Any] = load_config(Path("./config.yaml"))
    
    # embedding configs
    device: str = config["embedding"]["device"]
    backend: str = config["embedding"]["backend"]

    # chunking configs
    skip_batches: int = config["chunking"]["skip_batches"]
    if skip_batches > 0:
        warnings.warn(message="Batch skipping enabled...", category=UserWarning)

    chunk_type: str = config["chunking"]["type"]
    breakpoint_threshold_amount: int = config["chunking"]["percentile"]
    
    if chunk_type != "semantic":
        # only used for RecursiveCharacterTextSplitter
        chunk_size: int = config["chunking"]["chunk_size"]
        overlap: int = config["chunking"]["overlap"]

    # vector db configs
    vector_db_collection_name: str = config["vectordb"]["collection_name"]
    vector_db_save_path: Path = Path(config["vectordb"]["save_path"])

    unique_titles = get_unique_titles_from_squad(split="all")
    wiki_ds: IterableDataset = get_wiki(
        accepted_titles=unique_titles
    )
    embedding_model = Embedder(
        device=device, backend=backend
    )
    if chunk_type == "semantic":
        chunker: SemanticChunker = SemanticChunkerSingleton(
            embedding_model,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )
    else:
        chunker: RecursiveCharacterTextSplitter = RecursiveChunkerSingleton(
            chunk_size=chunk_size, chunk_overlap=overlap,
        )
    
    vector_db = WikipediaVectorStore(
        embedder=embedding_model,
        db_save_path=vector_db_save_path,
        collection_name=vector_db_collection_name,
    )

    create_vector_db(
        wiki_ds, unique_titles, chunker, vector_db, skip_batches,
    )


if __name__ == "__main__":
    main()
