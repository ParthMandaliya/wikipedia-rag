from typing import List, Literal
from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    """E5-small-v2 embedder with asymmetric encoding using HuggingFaceEmbeddings"""
    
    _instance = None
    _model = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        backend: Literal["torch", "onnx", "openvino"],
        batch_size: int = 128,
    ) -> None:
        if Embedder._model is None:
            self.batch_size = batch_size
            Embedder._model = HuggingFaceEmbeddings(
                model_name="intfloat/e5-small-v2",
                # multi_process has pickling issue with this embedding
                multi_process=False,
                model_kwargs={
                    "device": device,
                    "backend": backend,
                },
                encode_kwargs={
                    "normalize_embeddings": True, 
                    "batch_size": batch_size,
                },
                query_encode_kwargs={
                    "normalize_embeddings": True,
                }
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with passage prefix"""
        return Embedder._model.embed_documents([f"passage: {text}" for text in texts])
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with query prefix"""
        return Embedder._model.embed_query(f"query: {text}")
    