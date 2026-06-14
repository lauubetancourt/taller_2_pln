"""Embeddings semanticos con Sentence-Transformers."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .config import BATCH_SIZE_SENTENCE_EMBEDDINGS

SENTENCE_EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "BAAI/bge-m3",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def load_sentence_embedding_model(model_name: str) -> Any:
    """Cargar un modelo de Sentence-Transformers por nombre."""

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    setattr(model, "_taller_model_name", model_name)
    return model


def encode_texts(
    texts: list[str],
    model: Any,
    batch_size: int = BATCH_SIZE_SENTENCE_EMBEDDINGS,
    normalize_embeddings: bool = True,
    model_name: str | None = None,
    input_type: str = "passage",
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Generar embeddings para una lista de textos por lotes."""

    if not texts:
        return np.empty((0, 0), dtype="float32")

    resolved_model_name = model_name or getattr(model, "_taller_model_name", "")
    formatted_texts = [
        format_text_for_model(text, resolved_model_name, input_type=input_type)
        for text in texts
    ]

    embeddings = model.encode(
        formatted_texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(embeddings)


def encode_query_and_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    model: Any,
    batch_size: int = BATCH_SIZE_SENTENCE_EMBEDDINGS,
) -> tuple[np.ndarray, np.ndarray]:
    """Generar embeddings para una consulta y una coleccion de fragmentos."""

    chunk_texts = [extract_chunk_text(chunk) for chunk in chunks]
    query_embedding = encode_texts(
        [query],
        model,
        batch_size=batch_size,
        input_type="query",
    )
    chunk_embeddings = encode_texts(
        chunk_texts,
        model,
        batch_size=batch_size,
        input_type="passage",
        show_progress_bar=True,
    )
    return query_embedding, chunk_embeddings


def extract_chunk_text(chunk: dict[str, Any]) -> str:
    """Extraer el texto principal de un fragmento persistido."""

    text = chunk.get("text") or chunk.get("page_content")
    if not text:
        raise ValueError(f"El fragmento no contiene texto: {chunk}")
    return str(text)


def format_text_for_model(
    text: str, model_name: str, input_type: str = "passage"
) -> str:
    """Aplicar convenciones ligeras requeridas por algunos modelos."""

    clean_text = str(text).strip()
    if _requires_e5_prefix(model_name):
        lowered = clean_text.lower()
        if lowered.startswith("query: ") or lowered.startswith("passage: "):
            return clean_text
        prefix = "query: " if input_type == "query" else "passage: "
        return f"{prefix}{clean_text}"
    return clean_text


def model_name_to_slug(model_name: str) -> str:
    """Convertir el nombre de un modelo en un identificador seguro para archivos."""

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_")
    return slug.replace("/", "_")


def _requires_e5_prefix(model_name: str) -> bool:
    normalized = model_name.lower()
    return normalized.startswith("intfloat/multilingual-e5") or "/e5-" in normalized
