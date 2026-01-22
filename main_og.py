from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

from datasets import DatasetDict
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from data import get_wiki
from embeddings import Embedder
from data.preprocessing import SemanticChunkerSingleton, get_chunks
from data.preprocessing import RecursiveChunkerSingleton, get_non_semantic_chunks
from vector_db import WikipediaVectorStore
from utilities import load_config


def create_vector_db(
    wiki_ds: DatasetDict,
    chunker: SemanticChunker,
    vector_db: WikipediaVectorStore,
    batch_size: int = 128,
) -> None:
    with tqdm(desc="Chunking dataset", total=None) as pbar:
        docs_to_split: List[Document] = []
        for i, article in enumerate(wiki_ds):
            doc = Document(
                page_content=article["text"],
                metadata={
                    "article_id": article.get("id", ""),
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                }
            )
            pbar.set_description(f"Downloading rows {i:,}...")
            docs_to_split.append(doc)

            if len(docs_to_split) >= batch_size:
                pbar.set_description(f"Splitting {len(docs_to_split):,} rows into chunks...")
                splitted_docs: List[Document] = get_chunks(docs_to_split, chunker)
                docs_to_split.clear()
                
                pbar.set_description(
                    f"Updating vector store with {len(splitted_docs):,} chunked documents..."
                )
                vector_db.add_documents(splitted_docs)
                del splitted_docs

                pbar.update(batch_size)
        
        pbar.set_description("Dataset split, embedding & vector store is done")

    vector_store_save_path: Path = Path("./vector_store/store.faiss")
    vector_db.save(vector_store_save_path)
    print(f"Vector store stored at: {vector_store_save_path}")

def main():
    config: Dict[str, Dict[str, Any]] = load_config(Path("./config.json"))
    
    # embedding configs
    batch_size: int = config["embedding"]["batch_size"]
    device: str = config["embedding"]["device"]
    backend: str = config["embedding"]["backend"]

    # chunking configs
    chunk_type: str = config["chunking"]["type"]
    breakpoint_threshold_amount: int = config["chunking"]["percentile"]
    chunk_size: int = config["chunking"]["chunk_size"]
    overlap: int = config["chunking"]["overlap"]

    # vector db configs
    vector_db_type: str = config["vectordb"]["type"]
    vector_db_save_path: str = config["vectordb"]["save_path"]

    wiki_ds = get_wiki()
    embedding_model = Embedder(batch_size=batch_size, device=device, backend=backend)
    if chunk_type == "semantic":
        chunker: SemanticChunker = SemanticChunkerSingleton(
            embedding_model,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )
    else:
        chunker: RecursiveChunkerSingleton = RecursiveChunkerSingleton(
            chunk_size=chunk_size, chunk_overlap=overlap,
        )
    
    vector_db = WikipediaVectorStore(
        embedder=embedding_model,
        db_type=vector_db_type,
        db_save_path=vector_db_save_path,
    )

    create_vector_db(wiki_ds, chunker, vector_db, batch_size)


if __name__ == "__main__":
    main()
