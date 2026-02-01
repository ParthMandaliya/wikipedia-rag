import warnings
from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Set

from datasets import IterableDataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from data import get_hotpotqa
from embeddings import Embedder
from utilities import generate_article_uuid
from data.preprocessing import (
    SemanticChunkerSingleton, RecursiveChunkerSingleton, get_chunks
)
from vector_db import WikipediaVectorStore
from utilities import load_config


def create_vector_db(
    hotpotqad_ds: IterableDataset,
    chunker: Union[SemanticChunker, RecursiveCharacterTextSplitter],
    vector_db: WikipediaVectorStore,
    skip_n_articles: Optional[int] = 0,
) -> None:
    with tqdm(desc="Downloading...", total=None) as pbar:
        articles_processed: Set[str] = set()
        article_skipped: int = 0
        for i, article in enumerate(hotpotqad_ds):
            if article_skipped < skip_n_articles:
                for row_dict in article["full_articles"]:
                    articles_processed.add(row_dict["title"])

                article_skipped = len(articles_processed)
                pbar.set_description(
                    f"Skipping row: {i:,}; "
                    f"Articles skipped: {article_skipped:,}"
                )
                continue
            
            pbar.set_description(
                f"Rows: {i:,} | Articles: {len(articles_processed):,} | "
                f"Chunks: {vector_db.total_chunks:,} | Downloading..."
            )

            docs_to_split: List[Document] = []
            for full_article in article["full_articles"]:
                if full_article["title"] in articles_processed:
                    continue
                
                title = full_article["title"]
                text = full_article["article"]
                article_id = generate_article_uuid(title=title, text=text)
                doc = Document(
                    page_content=text,
                    metadata={
                        "article_id": article_id,
                        "title": title,
                    }
                )
                articles_processed.add(title)
                docs_to_split.append(doc)
            
            if len(docs_to_split) <= 0:
                continue

            # using loop to maintain chunk_index and total_chunks 
            # metadata per article
            splitted_docs: List[Document] = []
            for j, doc in enumerate(docs_to_split):
                pbar.set_description(
                    f"Rows: {i:,} | Articles: {len(articles_processed):,} | "
                    f"Chunks: {vector_db.total_chunks:,} | "
                    f"Chunking {j:,}/{len(docs_to_split)} documents..."
                )
                splitted_docs.extend(get_chunks(
                    [doc], chunker=chunker,
                ))
            
            pbar.set_description(
                f"Rows: {i:,} | Articles: {len(articles_processed):,} | "
                f"Chunks: {vector_db.total_chunks:,} | "
                f"Adding {len(splitted_docs):,} chunks to ChromaDB..."
            )
            vector_db.add_documents(splitted_docs)

            pbar.set_description(
                f"Rows: {i:,} | Articles: {len(articles_processed):,} | "
                f"Chunks: {vector_db.total_chunks:,} | "
                f"Added {len(splitted_docs):,} chunks to ChromaDB..."
            )
            del splitted_docs

            pbar.update(len(docs_to_split))
        
        pbar.set_description(
            f"Complete! Unique Articles Processed: {len(articles_processed):,} | "
            f"Total Rows processed: {i:,} | "
            f"Total Chunks Stored in ChromaDB: {vector_db.total_chunks:,}"
        )

    print(f"Final vector store saved at: {vector_db.db_save_path}")
    print(f"Total unique articles processed: {len(articles_processed):,}")
    print(f"Total rows processed: {i+1:,}")
    print(f"Total chunks indexed: {vector_db.total_chunks:,}")


def main():
    config: Dict[str, Any] = load_config(Path("./config.yaml"))
    
    # embedding configs
    device: str = config["embedding"]["device"]
    backend: str = config["embedding"]["backend"]

    # chunking configs
    skip_articles: int = config["chunking"]["skip_articles"]
    if skip_articles > 0:
        warnings.warn(message="Batch skipping enabled...", category=UserWarning)

    chunk_type: str = config["chunking"]["type"]
    breakpoint_threshold_amount: int = config["chunking"]["percentile"]
    
    # vector db configs
    vector_db_collection_name: str = config["vectordb"]["collection_name"]
    vector_db_save_path: Path = Path(config["vectordb"]["save_path"])

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

    create_vector_db(
        hotpotqa_ds, chunker, vector_db, skip_articles,
    )
 

if __name__ == "__main__":
    main()
