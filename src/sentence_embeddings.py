"""Embeddings semanticos con Sentence-Transformers."""

from __future__ import annotations

from typing import Any

from .config import BATCH_SIZE_SENTENCE_EMBEDDINGS


SENTENCE_EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "BAAI/bge-m3",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def load_sentence_embedding_model(model_name: str) -> Any:
    """Cargar un modelo de Sentence-Transformers por nombre."""

    raise NotImplementedError("Pendiente: cargar modelo de embeddings semanticos.")


def encode_texts(
    texts: list[str],
    model: Any,
    batch_size: int = BATCH_SIZE_SENTENCE_EMBEDDINGS,
    normalize_embeddings: bool = True,
) -> Any:
    """Generar embeddings para una lista de textos por lotes."""

    raise NotImplementedError("Pendiente: generar embeddings de textos.")


def encode_query_and_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    model: Any,
    batch_size: int = BATCH_SIZE_SENTENCE_EMBEDDINGS,
) -> tuple[Any, Any]:
    """Generar embeddings para una consulta y una coleccion de fragmentos."""

    raise NotImplementedError("Pendiente: codificar consulta y fragmentos.")
