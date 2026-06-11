"""Calculo de similitud semantica entre consultas y fragmentos."""

from __future__ import annotations

from typing import Any


def cosine_similarity_scores(query_embedding: Any, chunk_embeddings: Any) -> Any:
    """Calcular similitud coseno entre una consulta y multiples fragmentos."""

    raise NotImplementedError("Pendiente: calcular similitudes coseno.")


def top_k_similar_chunks(
    chunks: list[dict[str, Any]],
    scores: Any,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Devolver los ``k`` fragmentos mas similares con sus puntajes."""

    raise NotImplementedError("Pendiente: seleccionar top-k fragmentos.")


def find_most_similar_chunk(chunks: list[dict[str, Any]], scores: Any) -> dict[str, Any]:
    """Identificar el fragmento mas similar a una consulta."""

    raise NotImplementedError("Pendiente: obtener fragmento mas similar.")
