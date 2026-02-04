from tqdm import tqdm
from typing import Union, Optional, Set, List

from datasets import IterableDataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from utilities import generate_article_uuid
from vector_db import WikipediaVectorStore
from data.preprocessing import get_chunks


def create_vector_db(
    hotpotqad_ds: IterableDataset,
    chunking_type: str,
    chunker: Union[SemanticChunker, RecursiveCharacterTextSplitter],
    vector_db: WikipediaVectorStore,
    skip_n_articles: int = 0,
    process_n_articles: Optional[int] = None,
) -> None:
    with tqdm(desc="Downloading...", total=None) as pbar:
        articles_processed: Set[str] = set()
        article_skipped: int = 0
        for i, article in enumerate(hotpotqad_ds):
            if article_skipped < skip_n_articles:
                cont = True
                for row_dict in article["full_articles"]:
                    articles_processed.add(row_dict["title"])

                    if len(articles_processed) >= skip_n_articles:
                        cont = False
                        break

                article_skipped = len(articles_processed)
                pbar.set_description(
                    f"{chunking_type.title()} | "
                    f"Skipping row: {i:,}; "
                    f"Articles skipped: {article_skipped:,}"
                )
                if cont:
                    continue
            
            pbar.set_description(
                f"{chunking_type.title()} | "
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
                    f"{chunking_type.title()} | "
                    f"Rows: {i:,} | Articles: {len(articles_processed):,} | "
                    f"Chunks: {vector_db.total_chunks:,} | "
                    f"Chunking {j:,}/{len(docs_to_split)} documents..."
                )
                splitted_docs.extend(get_chunks(
                    [doc], chunker=chunker,
                ))
            
            pbar.set_description(
                f"{chunking_type.title()} | "
                f"Rows: {i:,} | Articles: {len(articles_processed):,} | "
                f"Chunks: {vector_db.total_chunks:,} | "
                f"Adding {len(splitted_docs):,} chunks to ChromaDB..."
            )
            vector_db.add_documents(splitted_docs)

            pbar.set_description(
                f"{chunking_type.title()} | "
                f"Rows: {i:,} | Articles: {len(articles_processed):,} | "
                f"Chunks: {vector_db.total_chunks:,} | "
                f"Added {len(splitted_docs):,} chunks to ChromaDB..."
            )
            del splitted_docs

            pbar.update(len(docs_to_split))

            if (
                (process_n_articles is not None) and 
                len(articles_processed) >= process_n_articles
            ):
                break
        
        pbar.set_description(
            f"{chunking_type.title()} | "
            f"Complete! Unique Articles Processed: {len(articles_processed):,} | "
            f"Total Unique Articles Skipped: {article_skipped:,} | "
            f"Total Rows processed: {i+1:,} | "
            f"Total Chunks Stored in ChromaDB: {vector_db.total_chunks:,}"
        )

    print(f"{chunking_type.title()}")
    print(f"Final vector store saved at: {vector_db.db_save_path}")
    print(f"Total unique articles skipped: {article_skipped:,}")
    print(f"Total unique articles processed: {len(articles_processed):,}")
    print(f"Total rows processed: {i+1:,}")
    print(f"Total chunks indexed: {vector_db.total_chunks:,}")
