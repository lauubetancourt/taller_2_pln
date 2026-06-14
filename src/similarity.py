"""Calculo de similitud semantica entre consultas y fragmentos."""

from __future__ import annotations

from typing import Any

import numpy as np


def cosine_similarity_scores(query_embedding: Any, chunk_embeddings: Any) -> np.ndarray:
    """Calcular similitud coseno entre una consulta y multiples fragmentos."""

    query = _as_2d_array(query_embedding)
    chunks = _as_2d_array(chunk_embeddings)

    if query.shape[0] != 1:
        raise ValueError(
            "query_embedding debe representar una sola consulta "
            f"y tiene forma {query.shape}."
        )
    if chunks.shape[0] == 0:
        return np.empty((0,), dtype="float32")
    if query.shape[1] != chunks.shape[1]:
        raise ValueError(
            "Las dimensiones de la consulta y los fragmentos no coinciden: "
            f"{query.shape[1]} != {chunks.shape[1]}."
        )

    normalized_query = _l2_normalize(query)
    normalized_chunks = _l2_normalize(chunks)
    return normalized_chunks @ normalized_query[0]


def top_k_similar_chunks(
    chunks: list[dict[str, Any]],
    scores: Any,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Devolver los ``k`` fragmentos mas similares con sus puntajes."""

    score_array = np.asarray(scores, dtype="float32").reshape(-1)
    if len(score_array) != len(chunks):
        raise ValueError(
            "La cantidad de scores no coincide con la cantidad de fragmentos: "
            f"{len(score_array)} != {len(chunks)}."
        )

    top_indices = np.argsort(score_array)[::-1][: max(0, k)]
    results: list[dict[str, Any]] = []
    for rank, chunk_position in enumerate(top_indices, start=1):
        enriched_chunk = dict(chunks[int(chunk_position)])
        enriched_chunk["chunk_position"] = int(chunk_position)
        enriched_chunk["similarity_score"] = float(score_array[int(chunk_position)])
        enriched_chunk["rank"] = rank
        results.append(enriched_chunk)
    return results


def find_most_similar_chunk(
    chunks: list[dict[str, Any]], scores: Any
) -> dict[str, Any]:
    """Identificar el fragmento mas similar a una consulta."""

    results = top_k_similar_chunks(chunks, scores, k=1)
    if not results:
        raise ValueError("No hay fragmentos para seleccionar el mas similar.")
    return results[0]


def _as_2d_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype="float32")
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(
            f"Se esperaba un arreglo 1D o 2D, se recibio forma {array.shape}."
        )
    return array


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return matrix / safe_norms
