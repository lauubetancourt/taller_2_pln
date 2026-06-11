"""Entrenamiento y analisis de Word2Vec y FastText con Gensim."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import N_WORKERS


def build_gensim_corpus(sentences: Iterable[str]) -> Iterable[list[str]]:
    """Convertir sentencias crudas en un corpus iterable de listas de tokens."""

    raise NotImplementedError("Pendiente: construir corpus para Gensim.")


def train_word2vec_model(
    corpus: Iterable[list[str]],
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 5,
    workers: int = N_WORKERS,
) -> Any:
    """Entrenar un modelo Word2Vec con parametros base del taller."""

    raise NotImplementedError("Pendiente: entrenar Word2Vec.")


def train_fasttext_model(
    corpus: Iterable[list[str]],
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 5,
    workers: int = N_WORKERS,
) -> Any:
    """Entrenar un modelo FastText con parametros base del taller."""

    raise NotImplementedError("Pendiente: entrenar FastText.")


def save_embedding_model(model: Any, output_path: str | Path) -> None:
    """Guardar un modelo de embeddings entrenado en disco."""

    raise NotImplementedError("Pendiente: guardar modelo de embeddings.")


def get_similar_words(model: Any, word: str, topn: int = 10) -> list[tuple[str, float]]:
    """Obtener palabras similares a partir de un modelo Gensim."""

    raise NotImplementedError("Pendiente: consultar palabras similares.")


def get_fasttext_oov_vector(model: Any, word: str) -> Any:
    """Obtener vector para palabra OOV usando subpalabras de FastText."""

    raise NotImplementedError("Pendiente: manejar OOV con FastText.")
