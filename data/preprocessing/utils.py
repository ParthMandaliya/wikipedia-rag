from typing import List, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from utilities import generate_uuid


def get_chunks(
    docs: List[Document],
    chunker: Union[SemanticChunker, RecursiveCharacterTextSplitter],
) -> List[Document]:
    chunked_documents: List[Document] = []

    chunked_docs: List[Document] = chunker.create_documents(
        [doc.page_content for doc in docs],
        [doc.metadata for doc in docs]
    )
    for i, chunked_doc in enumerate(chunked_docs):
        chunk_uuid: str = generate_uuid(
            article_id=chunked_doc.metadata["article_id"],
            chunk_index=i,
            text=chunked_doc.page_content
        )
        chunked_doc.metadata.update({
            "chunk_uuid": chunk_uuid,
            "chunk_index": i,
            "total_chunks": len(chunked_docs),
        })
        chunked_documents.append(chunked_doc)
    
    return chunked_documents
