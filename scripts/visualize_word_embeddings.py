"""Visualizar embeddings de palabras con PCA y t-SNE."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import WORD_EMBEDDINGS_FIGURES_DIR, WORD_EMBEDDINGS_OUTPUT_DIR
from src.visualization import (
    create_reduction_dataframe,
    get_vectors_for_words,
    plot_2d_words,
    reduce_embeddings_pca,
    reduce_embeddings_tsne,
    select_words_for_visualization,
)
from src.word_embeddings import load_gensim_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word2vec-model", required=True)
    parser.add_argument("--fasttext-model", required=True)
    parser.add_argument("--training-size-label", required=True)
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--preferred-words", default=None)
    parser.add_argument("--run-tsne", action="store_true")
    parser.add_argument("--run-pca", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    WORD_EMBEDDINGS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    WORD_EMBEDDINGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_pca = args.run_pca or not args.run_tsne
    run_tsne = args.run_tsne
    preferred_words = _parse_preferred_words(args.preferred_words)

    models = [
        ("word2vec", load_gensim_model(args.word2vec_model)),
        ("fasttext", load_gensim_model(args.fasttext_model)),
    ]

    for model_name, model in models:
        words = select_words_for_visualization(
            model,
            max_words=args.max_words,
            preferred_words=preferred_words,
        )
        vectors, valid_words = get_vectors_for_words(model, words)
        if len(valid_words) < 2:
            print(f"{model_name}: no hay suficientes palabras para visualizar.")
            continue

        if run_pca:
            _save_reduction(
                vectors,
                valid_words,
                method="pca",
                model_name=model_name,
                training_size=args.training_size_label,
            )
        if run_tsne:
            _save_reduction(
                vectors,
                valid_words,
                method="tsne",
                model_name=model_name,
                training_size=args.training_size_label,
            )

    print("\nInterpretacion breve:")
    print("- PCA ayuda a observar estructura global aproximada.")
    print("- t-SNE ayuda a observar vecindarios locales.")
    print("- t-SNE no debe leerse como distancia global exacta entre todos los puntos.")


def _save_reduction(
    vectors,
    words: list[str],
    method: str,
    model_name: str,
    training_size: str,
) -> None:
    if method == "pca":
        points = reduce_embeddings_pca(vectors)
    elif method == "tsne":
        points = reduce_embeddings_tsne(vectors)
    else:
        raise ValueError(f"Metodo no soportado: {method}")

    title = f"{model_name} {method.upper()} ({training_size})"
    figure_path = WORD_EMBEDDINGS_FIGURES_DIR / f"{model_name}_{method}_{training_size}.png"
    coordinates_path = WORD_EMBEDDINGS_OUTPUT_DIR / f"{model_name}_{method}_{training_size}.csv"

    plot_2d_words(points, words, title=title, output_path=figure_path)
    coordinates_df = create_reduction_dataframe(
        words,
        points,
        method=method,
        model_name=model_name,
        training_size=training_size,
    )
    coordinates_df.to_csv(coordinates_path, index=False)
    print(f"Figura guardada: {figure_path}")
    print(f"Coordenadas guardadas: {coordinates_path}")


def _parse_preferred_words(value: str | None) -> list[str]:
    if not value:
        return []
    return [word.strip() for word in value.split(",") if word.strip()]


if __name__ == "__main__":
    main()
