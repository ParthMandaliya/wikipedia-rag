import warnings
from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Set

from datasets import DatasetDict
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
    wiki_ds: DatasetDict,
    accepted_titles: Set[str],
    chunker: Union[SemanticChunker, RecursiveCharacterTextSplitter],
    vector_db: WikipediaVectorStore,
    batch_size: Optional[int] = 128,
    skip_n_batches: Optional[int] = 0,
) -> None:
    with tqdm(desc="Downloading first batch...", total=None) as pbar:
        docs_to_split: List[Document] = []
        
        for batch_count, articles in enumerate(wiki_ds):
            if batch_count < skip_n_batches:
                pbar.set_description(f"Skipping batch: {batch_count:,}...")
                batch_count += 1
                pbar.update(batch_size)
                continue
            
            pbar.set_description(
                f"Articles: {batch_count:,} | Batches Processed: {batch_count:,} | Chunks: {vector_db.total_chunks:,} | "
                f"Downloading {batch_size:,} articles..."
            )

            keys = articles.keys()
            for j, row in enumerate(zip(*articles.values())):
                article = dict(zip(keys, row))
                doc = Document(
                    page_content=article["text"],
                    metadata={
                        "article_id": article.get("id", ""),
                        "url": article.get("url", ""),
                        "title": article.get("title", ""),
                    }
                )
                pbar.set_description(
                    f"Articles: {batch_count:,} | Batches Processed: {batch_count:,} | Chunks: {vector_db.total_chunks:,} | "
                    f"Processing article {j:,}..."
                )
                accepted_titles.remove(article.get("title", ""))
                docs_to_split.append(doc)

            pbar.set_description(
                f"Articles: {batch_count:,} | Batches Processed: {batch_count:,} | Chunks: {vector_db.total_chunks:,} | "
                f"Splitting {len(docs_to_split):,} docs..."
            )
            splitted_docs: List[Document] = get_chunks(docs_to_split, chunker)
            docs_to_split.clear()
            
            pbar.set_description(
                f"Articles: {batch_count:,} | Batches Processed: {batch_count:,} | Chunks: {vector_db.total_chunks:,} | "
                f"Adding {len(splitted_docs):,} chunks to ChromaDB..."
            )
            vector_db.add_documents(splitted_docs)
            batch_count += 1

            pbar.set_description(
                f"Articles: {batch_count:,} | Batches Processed: {batch_count:,} | Chunks: {vector_db.total_chunks:,} | "
                f"Added {len(splitted_docs):,} chunks to ChromaDB..."
            )
            del splitted_docs

            pbar.update(batch_size)

            if len(accepted_titles) <= 0:
                break
        
        pbar.set_description(
            f"Complete! Articles Processed: {batch_count+1:,} | Batches Processed: {batch_count:,} | Total Chunks Stored in ChromaDB: {vector_db.total_chunks:,}"
        )

    print(f"Final vector store saved at: {vector_db.db_save_path}")
    print(f"Total articles processed: {batch_count+1:,}")
    print(f"Total batches processed: {batch_count:,}")
    print(f"Total chunks indexed: {vector_db.total_chunks:,}")


def main():
    config: Dict[str, Any] = load_config(Path("./config.yaml"))
    
    # embedding configs
    batch_size: int = config["embedding"]["batch_size"]
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

    unique_titles = get_unique_titles_from_squad(split="both")
    wiki_ds: DatasetDict = get_wiki(
        batch_size=batch_size, drop_last_batch=False,
        accepted_titles=unique_titles
    )
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
        wiki_ds, unique_titles, chunker, vector_db, batch_size, skip_batches,
    )


if __name__ == "__main__":
    main()
