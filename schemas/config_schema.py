import os
import yaml
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic import model_validator, DirectoryPath


class Embedding(BaseModel):
    device: Literal["cuda", "cpu"] = Field(
        default="cpu",
        description="Device to load embedding model on, default: cpu"
    )
    backend: Literal["openvino", "torch", "onnx"] = Field(
        default="openvino",
        description="Backend to deploy the embedding model, default: openvino"
    )

    @model_validator(mode="after")
    def validate_backend(self) -> 'Embedding':
        if self.device == "cuda" and self.backend == "openvino":
            raise ValueError(f"{self.device} cannot be used with {self.backend} backend")
        return self

class Chunking(BaseModel):
    type: Literal["semantic", "recursive"] = Field(
        default="semantic",
        description="Type of chunking to use, default: semantic"
    )
    percentile: int = Field(
        default=70, gt=0, lt=100,
        description=(
            "Percentile threshold for semantic chunking breakpoints. "
            "Higher values (e.g., 85) create larger chunks with fewer splits; "
            "lower values (e.g., 50) create smaller, more granular chunks. "
            "Valid range: 1-99. Default: 70"
        )
    )
    chunk_size: int = Field(
        default=500, gt=0,
        description=(
            "Maximum chunk size to create, only used with recursive "
            "chunking approach"
        )
    )
    overlap: int = Field(
        default=100, gt=0,
        description=(
            "Number of overlapping characters between chunks to maintain "
            "context across boundaries. Only used with recursive chunking, "
            "default: 100"
        )
    )
    skip_articles: int = Field(
        default=0, ge=0,
        description=(
            "First n articles to skip, use this in case of resuming "
            "vector db creation, default: 0"
        )
    )
    process_articles: Optional[int] = Field(
        default=None, gt=0,
        description=(
            "Only process n articles instead of entire hotpotqa dataset "
            "default: None"
        )
    )

class VectorDb(BaseModel):
    create_vector_db: bool = Field(
        default=False,
        description="Whether to create vector db when running main.py or not"
    )
    collection_name: str = Field(
        default="wikipedia.en", 
        description="Collection name to use when creating/reading to/from ChromaDB"
    )
    db_path: Path = Field(
        default=f"{os.getcwd()}/vector_store",
        description="Save vector db to or read vector db from"
    )
    
    @field_validator("save_path", mode="before")
    @classmethod
    def validate_save_path(cls, value) -> Path:
        if isinstance(value, str):
            return Path(value)
        return value

    @model_validator(mode="after")
    def validate_save_path(self) -> 'VectorDb':
        if self.create_vector_db:
            if self.db_path.exists():
                if self.db_path.is_file():
                    raise ValueError(
                        f"{self.db_path} already exists as a file, not a directory"
                    )
                if len(list(self.db_path.iterdir())) > 0:
                    raise ValueError(
                        f"{self.db_path} already exists and is not empty"
                    )
            self.db_path.mkdir(parents=True, exist_ok=True)
        else:
            # Reading existing DB
            if not self.db_path.exists():
                raise ValueError(f"{self.db_path} does not exist")
            if not self.db_path.is_dir():
                raise ValueError(f"{self.db_path} is not a directory")
            if len(list(self.db_path.iterdir())) == 0:
                raise ValueError(f"{self.db_path} is empty")
        return self

class Config(BaseModel):
    embedding: Embedding
    chunking: Chunking
    vectordb: VectorDb

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            configs = yaml.safe_load(f)
        return cls(**configs)


if __name__ == "__main__":
    config = Config.from_yaml("/home/parth/env-rag-vdb-kg-py313/config.yaml")
    print()
    