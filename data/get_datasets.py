from typing import Literal
from datasets import load_dataset, DatasetDict


def get_wiki() -> DatasetDict:
    return load_dataset(
        "wikimedia/wikipedia", name="20231101.en",
        streaming=True, split="train",
    )

def get_squad_v2(split: str = Literal["train", "validation"]) -> DatasetDict:
    return load_dataset(
        "rajpurkar/squad_v2", split=split, streaming=True,
    )
