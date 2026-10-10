from typing import Any

import chromadb

from core.config import CHROMA_DIR


_PLACEHOLDER_EMBEDDING = [0.0]


def get_chroma_client() -> Any:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name: str = "idamp_memory") -> Any:
    return get_chroma_client().get_or_create_collection(
        name=name,
        embedding_function=None,
    )


def store_document(
    doc_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        collection = get_collection()
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
            embeddings=[_PLACEHOLDER_EMBEDDING],
        )
    except Exception:
        return


def query_memory(query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
    try:
        collection = get_collection()
        get_kwargs: dict[str, Any] = {
            "limit": n_results,
            "include": ["documents", "metadatas"],
        }
        if query_text:
            get_kwargs["where_document"] = {"$contains": query_text}

        results = collection.get(**get_kwargs)
        ids = results.get("ids", [])
        documents = results.get("documents") or [None] * len(ids)
        metadatas = results.get("metadatas") or [None] * len(ids)
        return [
            {
                "id": doc_id,
                "document": documents[index],
                "metadata": metadatas[index] or {},
            }
            for index, doc_id in enumerate(ids)
        ]
    except Exception:
        return []
