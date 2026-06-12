"""Analizar similaridad semantica y palabras OOV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import WORD_EMBEDDINGS_REPORTS_DIR
from src.word_embeddings import (
    fasttext_oov_similarity,
    get_most_similar_words,
    load_gensim_model,
)


DEFAULT_WORDS = ["gobierno", "presidente", "ciudad", "equipo", "mercado"]
DEFAULT_OOV_WORDS = ["hiperconectividad", "futbolísticamente"]
LIGHT_STOPWORDS = {
    "para",
    "como",
    "pero",
    "este",
    "esta",
    "sobre",
    "entre",
    "desde",
    "también",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word2vec-model", required=True)
    parser.add_argument("--fasttext-model", required=True)
    parser.add_argument("--words", default=None)
    parser.add_argument("--oov-words", default=None)
    parser.add_argument("--topn", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    WORD_EMBEDDINGS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    word2vec_model = load_gensim_model(args.word2vec_model)
    fasttext_model = load_gensim_model(args.fasttext_model)

    words = _parse_csv_arg(args.words) if args.words else _select_default_words(word2vec_model, fasttext_model)
    oov_words = _parse_csv_arg(args.oov_words) if args.oov_words else DEFAULT_OOV_WORDS
    common_words = [
        word
        for word in words
        if word in word2vec_model.wv.key_to_index and word in fasttext_model.wv.key_to_index
    ]
    if not common_words:
        raise ValueError("No hay palabras consultables presentes en ambos vocabularios.")

    word2vec_df = get_most_similar_words(word2vec_model, common_words, topn=args.topn)
    fasttext_df = get_most_similar_words(fasttext_model, common_words, topn=args.topn)
    similarity_df = pd.concat([word2vec_df, fasttext_df], ignore_index=True)

    oov_df = fasttext_oov_similarity(fasttext_model, oov_words, topn=args.topn)

    similarity_path = WORD_EMBEDDINGS_REPORTS_DIR / "similarity_analysis.csv"
    oov_path = WORD_EMBEDDINGS_REPORTS_DIR / "oov_fasttext_analysis.csv"
    similarity_df.to_csv(similarity_path, index=False)
    oov_df.to_csv(oov_path, index=False)

    print("=== Palabras analizadas ===")
    print(", ".join(common_words))
    print("\n=== Similaridad Word2Vec/FastText ===")
    print(similarity_df.head(30).to_string(index=False))
    print("\n=== OOV con FastText ===")
    print(oov_df.head(30).to_string(index=False))
    print(f"\nSimilaridad guardada: {similarity_path}")
    print(f"OOV guardado: {oov_path}")


def _parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _select_default_words(word2vec_model, fasttext_model, n_words: int = 5) -> list[str]:
    selected = [
        word
        for word in DEFAULT_WORDS
        if word in word2vec_model.wv.key_to_index and word in fasttext_model.wv.key_to_index
    ]
    for word in word2vec_model.wv.index_to_key:
        if len(selected) >= n_words:
            break
        if word in selected or word in LIGHT_STOPWORDS or len(word) < 4:
            continue
        if word in fasttext_model.wv.key_to_index and not any(char.isdigit() for char in word):
            selected.append(word)
    return selected[:n_words]


if __name__ == "__main__":
    main()
