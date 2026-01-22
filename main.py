from typing import Union, List, Dict, Any
from pathlib import Path
from tqdm import tqdm

from datasets import DatasetDict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from data import get_wiki
from embeddings import Embedder
from data.preprocessing import (
    SemanticChunkerSingleton, RecursiveChunkerSingleton, get_chunks
)
from vector_db import WikipediaVectorStore
from utilities import load_config


def create_vector_db(
    wiki_ds: DatasetDict,
    chunker: Union[SemanticChunker, RecursiveCharacterTextSplitter],
    vector_db: WikipediaVectorStore,
    batch_size: int = 128,
) -> None:
    with tqdm(desc="Chunking dataset", total=None) as pbar:
        docs_to_split: List[Document] = []
        batch_count: int = 0
        
        for i, article in enumerate(wiki_ds):
            doc = Document(
                page_content=article["text"],
                metadata={
                    "article_id": article.get("id", ""),
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                }
            )
            pbar.set_description(
                f"Articles: {i:,} | Batches: {batch_count:,} | Chunks: {vector_db.total_chunks:,} | "
                "Downloading..."
            )
            docs_to_split.append(doc)

            if len(docs_to_split) >= batch_size:
                batch_count += 1
                
                pbar.set_description(
                    f"Articles: {i:,} | Batches: {batch_count:,} | Chunks: {vector_db.total_chunks:,} | "
                    f"Splitting {len(docs_to_split):,} docs..."
                )
                splitted_docs: List[Document] = get_chunks(docs_to_split, chunker)
                docs_to_split.clear()
                
                pbar.set_description(
                    f"Articles: {i:,} | Batches: {batch_count:,} | Chunks: {vector_db.total_chunks:,} | "
                    f"Adding {len(splitted_docs):,} chunks to ChromaDB..."
                )
                vector_db.add_documents(splitted_docs)
                del splitted_docs

                pbar.update(batch_size)
                        
        # Handle remaining documents
        if docs_to_split:
            pbar.set_description(
                f"Articles: {i:,} | Chunks: {vector_db.total_chunks:,} | Processing final batch..."
            )
            splitted_docs: List[Document] = get_chunks(docs_to_split, chunker)
            vector_db.add_documents(splitted_docs)
            del splitted_docs
            docs_to_split.clear()
        
        pbar.set_description(
            f"Complete! Articles: {i+1:,} | Batches: {batch_count:,} | Total Chunks: {vector_db.total_chunks:,}"
        )

    print(f"Final vector store saved at: {vector_db.db_save_path}")
    print(f"Total articles processed: {i+1:,}")
    print(f"Total batches processed: {batch_count:,}")
    print(f"Total chunks indexed: {vector_db.total_chunks:,}")


def main():
    config: Dict[str, Any] = load_config(Path("./config.yaml"))
    
    # embedding configs
    batch_size: int = config["embedding"]["batch_size"]
    device: str = config["embedding"]["device"]
    backend: str = config["embedding"]["backend"]

    # chunking configs
    chunk_type: str = config["chunking"]["type"]
    breakpoint_threshold_amount: int = config["chunking"]["percentile"]
    
    if chunk_type != "semantic":
        # only used for RecursiveCharacterTextSplitter
        chunk_size: int = config["chunking"]["chunk_size"]
        overlap: int = config["chunking"]["overlap"]

    # vector db configs
    vector_db_collection_name: str = config["vectordb"]["collection_name"]
    vector_db_save_path: Path = Path(config["vectordb"]["save_path"])

    wiki_ds: DatasetDict = get_wiki()
    embedding_model = Embedder(batch_size=batch_size, device=device, backend=backend)
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
        wiki_ds, chunker, vector_db, batch_size
    )


if __name__ == "__main__":
    main()
