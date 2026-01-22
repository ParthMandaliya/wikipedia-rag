from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma

from embeddings import Embedder


class WikipediaVectorStore:
    def __init__(
        self,
        db_save_path: Path,
        collection_name: str,
        embedder: Optional[Embedder] = None
    ):
        self.embedder = embedder or Embedder()
        self.db_save_path = db_save_path
        self.db_save_path.mkdir(parents=True, exist_ok=True)

        self.vector_store: Chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embedder._model,
            persist_directory=self.db_save_path,
        )

    @property
    def total_chunks(self) -> int:
        return self.vector_store._collection.count()

    @property
    def max_batch_size(self) -> int:
        return self.vector_store._client.get_max_batch_size()

    def add_documents(self, documents: List[Document]):
        for i in range(0, len(documents), self.max_batch_size):
            batch = documents[i:i + self.max_batch_size]
            self.vector_store.add_documents(
                ids=[doc.metadata["chunk_uuid"] for doc in batch],
                documents=batch,
            )
        # self.vector_store.add_documents(
        #     ids=[doc.metadata["chunk_uuid"] for doc in documents],
        #     documents=documents,
        # )

    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Document]:
        if filter:
            return self.vector_store.similarity_search(
                query, k=k, filter=filter,
            )
        return self.vector_store.similarity_search(query, k=k)

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        if filter:
            return self.vector_store.similarity_search_with_score(
                query, k=k, filter=filter,
            )
        return self.vector_store.similarity_search_with_score(query, k=k,)

    @classmethod
    def load(
        cls,
        db_path: Path,
        embedder: Optional[Embedder] = None,
        collection_name: str = "wikipedia.en"
    ) -> 'WikipediaVectorStore':
        instance: 'WikipediaVectorStore' = cls(
            db_save_path=db_path,
            collection_name=collection_name,
            embedder=embedder,
        )
        
        print(f"Loaded Chroma collection from {db_path}")
        print(f"Total chunks: {instance.total_chunks}")
        
        return instance
