import uuid
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_filepath: Path) -> Dict[str, Any]:
    with open(config_filepath, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def save_config(config: Dict[str, Any], config_filepath: Path) -> None:
    with open(config_filepath, "w") as f:
        yaml.dump(
            config, f, Dumper=yaml.SafeDumper,
            indent=4, sort_keys=True
        )

def generate_article_uuid(title: str, text: str) -> str:
    return str(uuid.uuid5(
        uuid.NAMESPACE_DNS, f"{title}:{text}"
    ))

def generate_chunk_uuid(
    article_id: str, chunk_index: int, text: str,
) -> str:
    return str(uuid.uuid5(
        uuid.NAMESPACE_DNS, f"{article_id}:{chunk_index}:{text}"
    ))
