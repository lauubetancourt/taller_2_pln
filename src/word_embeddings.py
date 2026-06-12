"""Entrenamiento y analisis de Word2Vec y FastText con Gensim."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .config import (
    N_WORKERS,
    WORD_EMBEDDING_MIN_COUNT,
    WORD_EMBEDDING_VECTOR_SIZE,
    WORD_EMBEDDING_WINDOW,
)
from .preprocessing import tokenize_clean_sentence


def build_gensim_corpus(sentences: Iterable[str]) -> Iterable[list[str]]:
    """Convertir sentencias crudas en un corpus iterable de listas de tokens."""

    for sentence in sentences:
        tokens = tokenize_clean_sentence(sentence)
        if tokens:
            yield tokens


def train_word2vec(
    sentences: Iterable[list[str]],
    vector_size: int = WORD_EMBEDDING_VECTOR_SIZE,
    window: int = WORD_EMBEDDING_WINDOW,
    min_count: int = WORD_EMBEDDING_MIN_COUNT,
    workers: int = 4,
    sg: int = 1,
    epochs: int = 5,
) -> Any:
    """Entrenar un modelo Word2Vec con Gensim.

    ``sg=1`` activa Skip-Gram, una configuracion frecuente para capturar
    relaciones semanticas en corpus grandes.
    """

    from gensim.models import Word2Vec

    return Word2Vec(
        sentences=list(sentences),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )


def train_fasttext(
    sentences: Iterable[list[str]],
    vector_size: int = WORD_EMBEDDING_VECTOR_SIZE,
    window: int = WORD_EMBEDDING_WINDOW,
    min_count: int = WORD_EMBEDDING_MIN_COUNT,
    workers: int = 4,
    sg: int = 1,
    epochs: int = 5,
) -> Any:
    """Entrenar un modelo FastText con Gensim."""

    from gensim.models import FastText

    return FastText(
        sentences=list(sentences),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )


def save_model(model: Any, output_path: str | Path) -> None:
    """Guardar un modelo Gensim en disco."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def load_gensim_model(model_path: str | Path) -> Any:
    """Cargar Word2Vec o FastText desde disco."""

    from gensim.models import FastText, Word2Vec

    path = Path(model_path)
    try:
        return Word2Vec.load(str(path))
    except Exception:
        return FastText.load(str(path))


def get_vocabulary_stats(model: Any, topn: int = 30) -> dict[str, Any]:
    """Obtener estadisticas basicas del vocabulario de un modelo Gensim."""

    words = list(model.wv.index_to_key)
    top_words = []
    for word in words[:topn]:
        try:
            count = int(model.wv.get_vecattr(word, "count"))
        except KeyError:
            count = None
        top_words.append({"word": word, "count": count})

    return {
        "model_type": model.__class__.__name__,
        "vocab_size": len(words),
        "vector_size": int(model.wv.vector_size),
        "top_words": top_words,
    }


def save_embedding_matrix(
    model: Any,
    output_path: str | Path,
    max_words: int | None = None,
) -> tuple[Path, Path]:
    """Guardar matriz de embeddings ``.npy`` y vocabulario asociado ``.csv``."""

    base_path = Path(output_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path = base_path.with_suffix(".npy")
    vocab_path = base_path.with_suffix(".csv")

    words = list(model.wv.index_to_key)
    if max_words is not None:
        words = words[:max_words]
    matrix = np.vstack([model.wv[word] for word in words]) if words else np.empty((0, model.wv.vector_size))
    np.save(matrix_path, matrix)
    pd.DataFrame({"index": range(len(words)), "word": words}).to_csv(vocab_path, index=False)
    return matrix_path, vocab_path


def get_most_similar_words(
    model: Any,
    words: Sequence[str],
    topn: int = 10,
) -> pd.DataFrame:
    """Obtener palabras similares para una lista de consultas."""

    rows: list[dict[str, Any]] = []
    model_name = model.__class__.__name__
    for query_word in words:
        if query_word not in model.wv.key_to_index:
            rows.append(
                {
                    "modelo": model_name,
                    "palabra_consultada": query_word,
                    "palabra_similar": None,
                    "score": None,
                    "ranking": None,
                    "estado": "fuera_del_vocabulario",
                }
            )
            continue

        for rank, (similar_word, score) in enumerate(
            model.wv.most_similar(query_word, topn=topn),
            start=1,
        ):
            rows.append(
                {
                    "modelo": model_name,
                    "palabra_consultada": query_word,
                    "palabra_similar": similar_word,
                    "score": float(score),
                    "ranking": rank,
                    "estado": "ok",
                }
            )
    return pd.DataFrame(rows)


def fasttext_oov_similarity(
    model: Any,
    oov_words: Sequence[str],
    reference_words: Sequence[str] | None = None,
    topn: int = 10,
) -> pd.DataFrame:
    """Analizar OOV con FastText usando vectores de subpalabras."""

    rows: list[dict[str, Any]] = []
    explicit_vocab = set(model.wv.key_to_index)
    reference_set = set(reference_words or [])

    for query_word in oov_words:
        is_explicit_oov = query_word not in explicit_vocab
        try:
            vector = model.wv[query_word]
            similar_items = model.wv.similar_by_vector(vector, topn=topn + 5)
        except Exception as exc:
            rows.append(
                {
                    "modelo": model.__class__.__name__,
                    "palabra_oov": query_word,
                    "esta_en_vocabulario_explicito": not is_explicit_oov,
                    "palabra_similar": None,
                    "score": None,
                    "ranking": None,
                    "estado": f"error: {exc}",
                    "nota": "FastText intenta construir vectores OOV con n-gramas de caracteres.",
                }
            )
            continue

        rank = 1
        for similar_word, score in similar_items:
            if similar_word == query_word or (reference_set and similar_word not in reference_set):
                continue
            rows.append(
                {
                    "modelo": model.__class__.__name__,
                    "palabra_oov": query_word,
                    "esta_en_vocabulario_explicito": not is_explicit_oov,
                    "palabra_similar": similar_word,
                    "score": float(score),
                    "ranking": rank,
                    "estado": "ok",
                    "nota": "FastText puede generar vectores OOV con n-gramas de caracteres.",
                }
            )
            rank += 1
            if rank > topn:
                break
    return pd.DataFrame(rows)


def train_word2vec_model(
    corpus: Iterable[list[str]],
    vector_size: int = WORD_EMBEDDING_VECTOR_SIZE,
    window: int = WORD_EMBEDDING_WINDOW,
    min_count: int = WORD_EMBEDDING_MIN_COUNT,
    workers: int = N_WORKERS,
) -> Any:
    """Compatibilidad con el nombre inicial del proyecto."""

    return train_word2vec(corpus, vector_size, window, min_count, workers)


def train_fasttext_model(
    corpus: Iterable[list[str]],
    vector_size: int = WORD_EMBEDDING_VECTOR_SIZE,
    window: int = WORD_EMBEDDING_WINDOW,
    min_count: int = WORD_EMBEDDING_MIN_COUNT,
    workers: int = N_WORKERS,
) -> Any:
    """Compatibilidad con el nombre inicial del proyecto."""

    return train_fasttext(corpus, vector_size, window, min_count, workers)


def save_embedding_model(model: Any, output_path: str | Path) -> None:
    """Compatibilidad con el nombre inicial del proyecto."""

    save_model(model, output_path)


def get_similar_words(model: Any, word: str, topn: int = 10) -> list[tuple[str, float]]:
    """Obtener palabras similares a partir de un modelo Gensim."""

    return model.wv.most_similar(word, topn=topn)


def get_fasttext_oov_vector(model: Any, word: str) -> Any:
    """Obtener vector para palabra OOV usando subpalabras de FastText."""

    return model.wv[word]


def save_vocab_stats_csv(stats: list[dict[str, Any]], output_path: str | Path) -> None:
    """Guardar estadisticas de vocabulario de varios modelos en CSV."""

    rows = []
    for stat in stats:
        rows.append(
            {
                "model_type": stat["model_type"],
                "vocab_size": stat["vocab_size"],
                "vector_size": stat["vector_size"],
                "top_words_json": json.dumps(stat["top_words"], ensure_ascii=False),
            }
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
