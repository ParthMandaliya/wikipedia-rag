import uuid


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
