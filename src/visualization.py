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

    raise NotImplementedError("Pendiente: preparar subconjunto de embeddings.")


def reduce_embeddings_pca(embeddings: Any, n_components: int = 2, seed: int = 42) -> Any:
    """Reducir embeddings con PCA."""

    raise NotImplementedError("Pendiente: reducir con PCA.")


def reduce_embeddings_tsne(
    embeddings: Any,
    n_components: int = 2,
    perplexity: float = 30.0,
    seed: int = 42,
) -> Any:
    """Reducir embeddings con t-SNE."""

    raise NotImplementedError("Pendiente: reducir con t-SNE.")


def plot_2d_embeddings(
    coordinates: Any,
    labels: list[str] | None = None,
    title: str | None = None,
    output_path: str | None = None,
) -> Any:
    """Graficar coordenadas 2D y guardar la figura opcionalmente."""

    raise NotImplementedError("Pendiente: graficar embeddings 2D.")
