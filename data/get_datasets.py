import pandas as pd
from typing import Literal, List, Optional, Dict
from datasets import load_dataset, IterableDataset


def get_wiki(
    batch_size: int,
    accepted_titles: List[str],
    drop_last_batch: Optional[bool] = False
) -> IterableDataset:
    return load_dataset(
        "wikimedia/wikipedia", name="20231101.en",
        streaming=True, split="train",
    ).filter(
        lambda x: x["title"] in accepted_titles
    ).batch(
        batch_size=batch_size,
        drop_last_batch=drop_last_batch,
    )

def get_squad_v2(
    batch_size: int,
    split: str = Literal["train", "validation"],
    drop_last_batch: Optional[bool] = False
) -> IterableDataset:
    return load_dataset(
        "rajpurkar/squad_v2", split=split, streaming=True,
    ).batch(
        batch_size=batch_size,
        drop_last_batch=drop_last_batch,
    )

def get_unique_titles_from_squad(
    split: Literal["train", "validation", "both"],
    parquet_links: Optional[Dict[str, str]] = {
        "train": (
            "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/"
            "squad_v2/train-00000-of-00001.parquet"
        ),
        "val": (
            "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/"
            "squad_v2/validation-00000-of-00001.parquet"
        )
    }
) -> List[str]:
    if split != "both":
        return sorted(pd.read_parquet(parquet_links[split]).title.unique())
    elif split == "both":
        unique_titles = []
        for link in parquet_links.values():
            unique_titles.extend(pd.read_parquet(link).title.unique())
        return sorted(unique_titles)
    