import pandas as pd
from typing import Literal, Set, Optional, Dict
from datasets import load_dataset, IterableDataset


def get_wiki(
    accepted_titles: Set[str],
) -> IterableDataset:
    return load_dataset(
        "wikimedia/wikipedia", name="20231101.en",
        streaming=True, split="train",
    ).filter(
        lambda x: x["title"] in accepted_titles
    )

def get_squad_v2(
    split: Literal["train", "validation"],
) -> IterableDataset:
    return load_dataset(
        "rajpurkar/squad_v2", split=split, streaming=True,
    )

def get_unique_titles_from_squad(
    split: Literal["train", "validation", "all"],
    parquet_links: Optional[Dict[str, str]] = None
) -> Set[str]:
    if parquet_links is None:
        parquet_links = {
            "train": (
                "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/"
                "squad_v2/train-00000-of-00001.parquet"
            ),
            "validation": (
                "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/"
                "squad_v2/validation-00000-of-00001.parquet"
            )
        }
    if split == "all":
        unique_titles = []
        for link in parquet_links.values():
            unique_titles.extend(pd.read_parquet(link).title.unique())
        return set(unique_titles)
    else:
        return set(pd.read_parquet(parquet_links[split]).title.unique())
    