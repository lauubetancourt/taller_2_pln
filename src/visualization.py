"""Funciones de visualizacion del taller."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_fragmentation_ratios(df, output_path: str | Path | None = None) -> Any:
    """Graficar razones de fragmentacion WordPiece y SentencePiece por sentencia."""

    import matplotlib.pyplot as plt

    x_positions = range(len(df))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        x_positions,
        df["razon_fragmentacion_wordpiece_sin_especiales"],
        marker="o",
        label="WordPiece-BETO",
    )
    ax.plot(
        x_positions,
        df["razon_fragmentacion_sentencepiece_sin_especiales"],
        marker="s",
        label="SentencePiece-T5",
    )
    ax.set_title("Razon de fragmentacion sin tokens especiales")
    ax.set_xlabel("Sentencia")
    ax.set_ylabel("Subtokens / tokens originales")
    ax.set_xticks(list(x_positions))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)

    return fig, ax


def plot_unknown_counts(df, output_path: str | Path | None = None) -> Any:
    """Graficar conteos de ``<unk>`` por sentencia para ambos tokenizadores."""

    import matplotlib.pyplot as plt

    x_positions = list(range(len(df)))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [position - width / 2 for position in x_positions],
        df["n_unknown_wordpiece"],
        width=width,
        label="WordPiece-BETO",
    )
    ax.bar(
        [position + width / 2 for position in x_positions],
        df["n_unknown_sentencepiece"],
        width=width,
        label="SentencePiece-T5",
    )
    ax.set_title("Tokens desconocidos por sentencia")
    ax.set_xlabel("Sentencia")
    ax.set_ylabel("Numero de <unk>")
    ax.set_xticks(x_positions)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)

    return fig, ax


def prepare_embedding_subset(
    embeddings: Any,
    labels: list[str] | None = None,
    max_items: int = 5000,
) -> tuple[Any, list[str] | None]:
    """Preparar subconjuntos de embeddings para graficar.

    Los tamanos objetivo del taller incluyen 100, 5000 y 10000 sentencias.
    """

    subset = embeddings[:max_items]
    subset_labels = labels[:max_items] if labels is not None else None
    return subset, subset_labels


def select_words_for_visualization(
    model: Any,
    max_words: int = 200,
    preferred_words: list[str] | None = None,
) -> list[str]:
    """Seleccionar palabras interpretables para graficar embeddings."""

    selected: list[str] = []
    preferred_words = preferred_words or []
    for word in preferred_words:
        if word in model.wv.key_to_index and word not in selected:
            selected.append(word)

    for word in model.wv.index_to_key:
        if len(word) < 3:
            continue
        if any(character.isdigit() for character in word):
            continue
        if word not in selected:
            selected.append(word)
        if len(selected) >= max_words:
            break
    return selected


def get_vectors_for_words(model: Any, words: list[str]) -> tuple[Any, list[str]]:
    """Retornar matriz de vectores y palabras presentes en el modelo."""

    import numpy as np

    valid_words = [word for word in words if word in model.wv.key_to_index]
    vectors = (
        np.vstack([model.wv[word] for word in valid_words])
        if valid_words
        else np.empty((0, model.wv.vector_size))
    )
    return vectors, valid_words


def reduce_embeddings_pca(
    embeddings: Any,
    n_components: int = 2,
    random_state: int = 42,
    seed: int | None = None,
) -> Any:
    """Reducir embeddings con PCA."""

    from sklearn.decomposition import PCA

    state = random_state if seed is None else seed
    return PCA(n_components=n_components, random_state=state).fit_transform(embeddings)


def reduce_embeddings_tsne(
    embeddings: Any,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
    seed: int | None = None,
) -> Any:
    """Reducir embeddings con t-SNE."""

    from sklearn.manifold import TSNE

    n_samples = len(embeddings)
    if n_samples < 2:
        raise ValueError("t-SNE requiere al menos dos vectores.")
    adjusted_perplexity = min(perplexity, max(1, n_samples - 1))
    state = random_state if seed is None else seed
    return TSNE(
        n_components=n_components,
        perplexity=adjusted_perplexity,
        random_state=state,
        init="pca",
        learning_rate="auto",
    ).fit_transform(embeddings)


def plot_2d_words(
    points: Any,
    words: list[str],
    title: str,
    output_path: str | Path | None = None,
    max_labels: int = 80,
) -> Any:
    """Graficar palabras en 2D con matplotlib."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(points[:, 0], points[:, 1], s=24, alpha=0.75)
    for word, (x_coord, y_coord) in list(zip(words, points, strict=True))[:max_labels]:
        ax.annotate(word, (x_coord, y_coord), fontsize=8, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    return fig, ax


def create_reduction_dataframe(
    words: list[str],
    points: Any,
    method: str,
    model_name: str,
    training_size: str,
) -> Any:
    """Crear tabla con coordenadas reducidas para analisis posterior."""

    import pandas as pd

    return pd.DataFrame(
        {
            "word": words,
            "x": points[:, 0],
            "y": points[:, 1],
            "method": method,
            "model_name": model_name,
            "training_size": training_size,
        }
    )


def plot_2d_embeddings(
    coordinates: Any,
    labels: list[str] | None = None,
    title: str | None = None,
    output_path: str | None = None,
) -> Any:
    """Graficar coordenadas 2D y guardar la figura opcionalmente."""

    return plot_2d_words(
        coordinates,
        labels or [str(index) for index in range(len(coordinates))],
        title or "Embeddings 2D",
        output_path=output_path,
    )
