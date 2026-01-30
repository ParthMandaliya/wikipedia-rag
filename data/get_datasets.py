from typing import Literal, Tuple
from datasets import load_dataset, DatasetDict


def get_hotpotqa(
    name: Literal["fullwiki", "distractor"],
    split: Literal["train", "validation", "test"],
) -> DatasetDict:
    return load_dataset(
        "ParthMandaliya/hotpot_qa", name=name, split=split,
        streaming=True,
    )

def get_unique_titles_from_hotpotqa(
    name: Literal["fullwiki", "distractor"],
    split: Literal["train", "validation", "test", "all"],
) -> Tuple[str]:
    unique_titles = set()
    if split != "all":
        ds = get_hotpotqa(name=name, split=split)
        for row in ds:
            for title in row["context"]["title"]:
                unique_titles.add(title.strip())
        return tuple(unique_titles)
    else:
        splits = (
            ["train", "validation", "test"] 
            if name == "fullwiki" else 
            ["train", "validation"]
        )
        for split in splits:
            ds = get_hotpotqa(name=name, split=split)
            for row in ds:
                for title in row["context"]["title"]:
                    unique_titles.add(title.strip())
        return tuple(unique_titles)
    